from __future__ import annotations

from pathlib import Path

from autopaper.analysis.excerpt import PaperExcerpt, build_excerpt
from autopaper.analysis.reader import clip_paper_text, load_document
from autopaper.catalog import ensure_topic, resolve_authors
from autopaper.errors import AutoPaperError
from autopaper.llm.prompts import CLASSIFY_SYSTEM, SUGGEST_SYSTEM, classify_user_prompt, suggest_user_prompt
from autopaper.models import Author, Paper, ScoredPaper, Topic
from autopaper.storage.db import KnowledgeBase
from autopaper.textutil import collapse_ws, normalize_match


def _mentioned(name: str, text: str) -> bool:
    cleaned = collapse_ws(name)
    if len(cleaned) < 2:
        return False
    letters = cleaned.encode("ascii", errors="ignore").decode("ascii")
    if letters == cleaned and len(cleaned) < 3:
        return False
    return cleaned.casefold() in text.casefold()


def _topic_terms(topic: Topic) -> list[str]:
    return [topic.name, topic.query, *topic.keywords]


def suggest_catalog(text: str, authors: list[Author], topics: list[Topic], llm=None) -> dict:
    return suggest_many([build_excerpt("", text)], authors, topics, llm)["papers"][0]


def suggest_many(excerpts: list[PaperExcerpt], authors: list[Author], topics: list[Topic], llm=None) -> dict:
    rows = [_heuristic_excerpt(excerpt, authors, topics) for excerpt in excerpts]
    if llm is not None and excerpts and (authors or topics):
        rows = _merge_model_suggestions(rows, excerpts, authors, topics, llm)
    count = len(rows)
    return {"papers": rows, "message": f"已为 {count} 篇分别建议作者和主题。"}


def classify_topics(text: str, topics: list[Topic], llm=None) -> list[str]:
    matched = [topic.id for topic in topics if any(_mentioned(term, text) for term in _topic_terms(topic))]
    if llm is None or not topics:
        return matched
    try:
        payload = llm.complete_json(
            CLASSIFY_SYSTEM,
            classify_user_prompt(_topic_block(topics), clip_paper_text(text, 8000)),
            expect="object",
            max_tokens=1024,
        )
    except Exception:
        return matched
    if not isinstance(payload, dict):
        return matched
    return _known_ids(payload.get("topics"), {topic.id for topic in topics})


def reclassify_library(kb: KnowledgeBase, llm=None) -> dict:
    topics = kb.list_topics()
    papers = kb.list_local_papers()
    if not papers:
        raise AutoPaperError("没有本地论文，无法再分类。")
    links: list[dict] = []
    author_links: list[dict] = []
    for paper in papers:
        text = _local_text(paper)
        excerpt = build_excerpt(paper.title, text)
        brief = paper_brief(paper, excerpt)
        if collapse_ws(brief):
            chosen = classify_topics(brief, topics, llm)
            already = kb.direct_topic_ids(paper.key)
            by_id = {topic.id: topic for topic in kb.list_topics()}
            for topic_id in chosen:
                if topic_id in already:
                    continue
                topic = by_id.get(topic_id)
                if topic is None:
                    continue
                _link_topic(kb, paper, topic, "模型再分类" if llm is not None else "主题词匹配")
                already.add(topic_id)
                links.append({"paper": paper.title, "topic": topic.name})
        for name in excerpt.keywords[:2]:
            topic = ensure_topic(kb, name)
            if topic.id in kb.direct_topic_ids(paper.key):
                continue
            _link_topic(kb, paper, topic, "论文关键词")
            links.append({"paper": paper.title, "topic": topic.name})
        linked_authors = {author.id for author in kb.authors_of_paper(paper.key)}
        catalog = kb.list_authors()
        for name in excerpt.author_names[:2]:
            author = _match_author(name, catalog) or resolve_authors(kb, name)[0]
            catalog = kb.list_authors()
            if author.id in linked_authors:
                continue
            kb.link_author_paper(author.id, paper.key)
            linked_authors.add(author.id)
            author_links.append({"paper": paper.title, "author": author.name})
    if not links and not author_links:
        message = "已有论文都核对过了，没有需要新加的作者或主题。"
    else:
        message = f"已补上 {len(author_links)} 位作者、{len(links)} 条主题。默认写入一作、二作和前两个关键词。"
    return {
        "added": len(links),
        "authors_added": len(author_links),
        "links": links[:40],
        "authors": author_links[:40],
        "message": message,
    }


def paper_brief(paper: Paper, excerpt: PaperExcerpt | None = None, limit: int = 2200) -> str:
    if excerpt is None:
        excerpt = build_excerpt(paper.title, _local_text(paper))
    parts = [paper.title, paper.abstract, excerpt.topic_haystack()]
    if not excerpt.abstract and not excerpt.conclusion:
        parts.append(excerpt.front)
    return clip_paper_text("\n".join(part for part in parts if part), limit)


def _local_text(paper: Paper) -> str:
    path = Path(paper.local_path) if paper.local_path else None
    if path is None or not path.exists():
        return ""
    try:
        return load_document(path)
    except AutoPaperError:
        return ""


def _link_topic(kb: KnowledgeBase, paper: Paper, topic: Topic, reason: str) -> None:
    kb.link_topic_paper(
        topic.id,
        ScoredPaper(paper, 70, None, 70, reason, "candidate"),
        paper.key,
    )


def _heuristic_excerpt(excerpt: PaperExcerpt, authors: list[Author], topics: list[Topic]) -> dict:
    suggestion = _heuristic_suggestions(excerpt.author_haystack(), excerpt.topic_haystack(), authors, topics)
    suggestion["name"] = excerpt.name
    people = _recommend_people(excerpt)
    new_authors = []
    for person in people:
        match = _match_author(person["name"], authors)
        if match is not None:
            _add_hit(suggestion["authors"], match.id, match.name, person["role"], prefer=True)
        else:
            new_authors.append({**person, "selected": True, "reason": person["role"]})
    new_topics = []
    for index, keyword in enumerate(excerpt.keywords):
        selected = index < 2
        match = _match_topic(keyword, topics)
        if match is not None and selected:
            _add_hit(suggestion["topics"], match.id, match.name, "论文关键词")
        elif match is None:
            new_topics.append({"name": keyword, "selected": selected, "reason": "论文关键词"})
    suggestion["new_authors"] = new_authors
    suggestion["new_topics"] = new_topics
    suggestion["message"] = _suggestion_message(suggestion, people, excerpt.keywords)
    return suggestion


def _merge_model_suggestions(
    rows: list[dict],
    excerpts: list[PaperExcerpt],
    authors: list[Author],
    topics: list[Topic],
    llm,
) -> list[dict]:
    found: dict[str, dict] = {}
    for group in _groups(excerpts, 6):
        block = "\n\n".join(item.llm_text() for item in group)
        try:
            payload = llm.complete_json(
                SUGGEST_SYSTEM,
                suggest_user_prompt(_author_block(authors), _topic_block(topics), block),
                expect="object",
                max_tokens=2048,
            )
        except Exception:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
            continue
        allowed = {item.name for item in group}
        for row in payload["papers"]:
            if isinstance(row, dict) and str(row.get("name") or "") in allowed:
                found[str(row["name"])] = row
    merged = []
    by_author = {author.id: author for author in authors}
    by_topic = {topic.id: topic for topic in topics}
    author_ids = set(by_author)
    topic_ids = set(by_topic)
    for row, excerpt in zip(rows, excerpts):
        chosen = found.get(excerpt.name)
        if chosen is None:
            merged.append(row)
            continue
        reason = collapse_ws(str(chosen.get("reason") or "")) or "模型根据作者栏、摘要和结论挑选。"
        author_hits = list(row["authors"])
        seen_authors = {item["id"] for item in author_hits}
        for author_id in _known_ids(chosen.get("authors"), author_ids):
            if author_id not in seen_authors:
                author_hits.append({"id": author_id, "name": by_author[author_id].name, "reason": reason})
                seen_authors.add(author_id)
        topic_hits = list(row["topics"])
        seen_topics = {item["id"] for item in topic_hits}
        for topic_id in _known_ids(chosen.get("topics"), topic_ids):
            if topic_id not in seen_topics:
                topic_hits.append({"id": topic_id, "name": by_topic[topic_id].name, "reason": reason})
                seen_topics.add(topic_id)
        packed = _pack(author_hits, topic_hits)
        packed["name"] = excerpt.name
        packed["new_authors"] = row.get("new_authors") or []
        packed["new_topics"] = row.get("new_topics") or []
        packed["message"] = row.get("message") or packed["message"]
        merged.append(packed)
    return merged


def _groups(items: list[PaperExcerpt], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _heuristic_suggestions(author_text: str, topic_text: str, authors: list[Author], topics: list[Topic]) -> dict:
    author_hits = []
    for author in authors:
        names = [author.name, *author.aliases]
        if any(_mentioned(name, author_text) for name in names):
            author_hits.append({"id": author.id, "name": author.name, "reason": "姓名出现在作者栏或文件名里。"})
    topic_hits = []
    for topic in topics:
        if any(_mentioned(term, topic_text) for term in _topic_terms(topic)):
            topic_hits.append({"id": topic.id, "name": topic.name, "reason": "主题词出现在摘要、结论或文首里。"})
    return _pack(author_hits, topic_hits)


def _recommend_people(excerpt: PaperExcerpt) -> list[dict]:
    people = []
    names = list(excerpt.author_names)
    if names:
        people.append({"name": names[0], "role": "一作"})
    if len(names) > 1:
        people.append({"name": names[1], "role": "二作"})
    corresponding = excerpt.corresponding
    if corresponding:
        for person in people:
            if normalize_match(person["name"]) == normalize_match(corresponding):
                person["role"] = f"{person['role']} · 通信作者"
                corresponding = ""
                break
    if corresponding:
        people.append({"name": corresponding, "role": "通信作者"})
    return people


def _match_author(name: str, authors: list[Author]) -> Author | None:
    marker = normalize_match(name)
    for author in authors:
        labels = [author.name, author.id, *author.aliases]
        if marker in {normalize_match(label) for label in labels}:
            return author
    best: Author | None = None
    best_size = 1
    for author in authors:
        for label in [author.name, *author.aliases]:
            tokens = normalize_match(label).split()
            if len(tokens) > best_size and _tokens_in_order(marker, tokens):
                best = author
                best_size = len(tokens)
    return best


def _named_among(name: str, people: list[dict]) -> bool:
    marker = normalize_match(name)
    tokens = marker.split()
    for person in people:
        other = normalize_match(person["name"])
        if marker == other or _tokens_in_order(other, tokens) or _tokens_in_order(marker, other.split()):
            return True
    return False


def _tokens_in_order(name: str, tokens: list[str]) -> bool:
    haystack = name.split()
    index = 0
    for token in haystack:
        if index < len(tokens) and token == tokens[index]:
            index += 1
    return index == len(tokens)


def _match_topic(name: str, topics: list[Topic]) -> Topic | None:
    marker = normalize_match(name)
    for topic in topics:
        if marker in {normalize_match(topic.name), normalize_match(topic.id)}:
            return topic
    return None


def _add_hit(items: list[dict], item_id: str, name: str, reason: str, prefer: bool = False) -> None:
    for item in items:
        if item["id"] == item_id:
            if prefer:
                item["reason"] = reason
            return
    items.append({"id": item_id, "name": name, "reason": reason})


def _suggestion_message(suggestion: dict, people: list[dict], keywords: tuple[str, ...]) -> str:
    author_names = [f"{person['name']}（{person['role']}）" for person in people]
    for item in suggestion["authors"]:
        if not _named_among(item["name"], people):
            author_names.append(item["name"])
    topic_names = [item["name"] for item in suggestion["topics"]]
    known_topics = {normalize_match(name) for name in topic_names}
    for keyword in keywords:
        if normalize_match(keyword) not in known_topics:
            topic_names.append(keyword)
            known_topics.add(normalize_match(keyword))
    authors = "、".join(author_names) or "无"
    topics = "、".join(topic_names) or "无"
    return f"建议作者：{authors}。建议主题：{topics}。"


def _pack(authors: list[dict], topics: list[dict]) -> dict:
    author_names = "、".join(item["name"] for item in authors) or "无"
    topic_names = "、".join(item["name"] for item in topics) or "无"
    return {
        "authors": authors,
        "topics": topics,
        "message": f"建议作者：{author_names}。建议主题：{topic_names}。",
    }


def _author_block(authors: list[Author]) -> str:
    lines = []
    for author in authors:
        aliases = "、".join(author.aliases)
        extra = f"（别名：{aliases}）" if aliases else ""
        lines.append(f"- {author.id} {author.name}{extra}")
    return "\n".join(lines)


def _topic_block(topics: list[Topic]) -> str:
    lines = []
    for topic in topics:
        keywords = "、".join(topic.keywords)
        extra = f"（主题词：{keywords}）" if keywords else ""
        lines.append(f"- {topic.id} {topic.name}{extra}")
    return "\n".join(lines)


def _known_ids(raw, allowed: set[str]) -> list[str]:
    if not isinstance(raw, list):
        return []
    chosen = []
    seen: set[str] = set()
    for item in raw:
        topic_id = str(item or "").strip()
        if topic_id in allowed and topic_id not in seen:
            chosen.append(topic_id)
            seen.add(topic_id)
    return chosen
