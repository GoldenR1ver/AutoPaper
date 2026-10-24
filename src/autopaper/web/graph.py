from __future__ import annotations

from autopaper.analysis.notes import author_profile_path, file_updated, synthesis_path, topic_profile_path
from autopaper.catalog import format_author
from autopaper.models import Paper
from autopaper.storage.db import KnowledgeBase
from autopaper.textutil import collapse_ws

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


def _reading_bits(kb: KnowledgeBase, paper: Paper) -> dict:
    stored = kb.get_reading(paper.key)
    payload = stored["payload"] if stored else {}
    if not isinstance(payload, dict):
        payload = {}
    title = collapse_ws(str(payload.get("title") or "")) or paper.title
    one = collapse_ws(str(payload.get("one_sentence") or ""))
    field = collapse_ws(str(payload.get("field") or ""))
    problem = collapse_ws(str(payload.get("problem") or ""))
    method = collapse_ws(str(payload.get("method") or ""))
    profile = " ".join(part for part in (title, one, field, problem, method[:800], paper.abstract[:800]) if part)
    return {
        "title": title,
        "one_sentence": one,
        "has_note": bool(stored and (stored.get("markdown") or payload)),
        "note_updated_at": (stored or {}).get("created_at") or "",
        "profile": profile or paper.title,
    }


def _similarity(left: str, right: str) -> int:
    if not left or not right:
        return 0
    if fuzz is None:
        left_tokens = set(left.casefold().split())
        right_tokens = set(right.casefold().split())
        if not left_tokens or not right_tokens:
            return 0
        shared = len(left_tokens & right_tokens)
        return round(100 * shared / max(len(left_tokens), len(right_tokens)))
    return int(fuzz.token_set_ratio(left, right))


def build_library_graph(kb: KnowledgeBase, data_dir) -> dict:
    """论文、作者、主题，以及笔记文本算出的相关度。"""
    papers: dict[str, Paper] = {}
    topic_links: dict[str, list[dict]] = {}
    for topic in kb.list_topics():
        for scored in kb.list_topic_papers(topic.id):
            papers[scored.paper.key] = scored.paper
            topic_links.setdefault(scored.paper.key, []).append(
                {"id": topic.id, "name": topic.name, "score": round(float(scored.score), 1)}
            )
    for paper in kb.list_local_papers():
        papers.setdefault(paper.key, paper)

    readings = {key: _reading_bits(kb, paper) for key, paper in papers.items()}
    author_links: dict[str, list[dict]] = {}
    authors: dict[str, dict] = {}
    for paper in papers.values():
        linked = []
        for author in kb.authors_of_paper(paper.key):
            label = format_author(author)
            authors[author.id] = {
                "id": author.id,
                "name": author.name,
                "label": label,
                "aliases": list(author.aliases),
            }
            linked.append({"id": author.id, "label": label})
        author_links[paper.key] = linked

    known_authors = {item.id: item for item in kb.list_authors()}
    for author_id, record in authors.items():
        person = known_authors.get(author_id)
        profile = author_profile_path(data_dir, person) if person else None
        record["has_profile"] = bool(profile and profile.exists())
        record["profile_updated_at"] = file_updated(profile) if profile else ""
    known_topics = {item.id: item for item in kb.list_topics()}
    topics = {
        topic.id: {
            "id": topic.id,
            "name": topic.name,
            "has_profile": topic_profile_path(data_dir, topic).exists(),
            "profile_updated_at": file_updated(topic_profile_path(data_dir, topic)),
        }
        for topic in known_topics.values()
    }
    paper_nodes = []
    for paper in papers.values():
        reading = readings[paper.key]
        scores = [item["score"] for item in topic_links.get(paper.key, [])]
        paper_nodes.append(
            {
                "key": paper.key,
                "title": reading["title"],
                "stored_title": paper.title,
                "year": paper.year,
                "one_sentence": reading["one_sentence"],
                "has_note": reading["has_note"],
                "note_updated_at": reading["note_updated_at"],
                "has_file": bool(paper.local_path),
                "score": max(scores) if scores else 0,
                "authors": author_links.get(paper.key, []),
                "topics": topic_links.get(paper.key, []),
            }
        )

    keys = list(papers)
    edges = []
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            same_author = {item["id"] for item in author_links.get(left, [])} & {
                item["id"] for item in author_links.get(right, [])
            }
            same_topic = {item["id"] for item in topic_links.get(left, [])} & {
                item["id"] for item in topic_links.get(right, [])
            }
            if len(keys) > 80 and not same_author and not same_topic:
                continue
            weight = _similarity(readings[left]["profile"], readings[right]["profile"])
            if weight < 20 and not same_author and not same_topic:
                continue
            if weight >= 55:
                reason = "笔记相近"
            elif same_author:
                reason = "同一作者"
            elif same_topic:
                reason = "同一主题"
            else:
                reason = "文本相近"
            edges.append({"source": left, "target": right, "weight": weight, "reason": reason})

    syntheses = []
    seen_pairs: set[tuple[str, str]] = set()
    for paper in papers.values():
        for author_item in author_links.get(paper.key, []):
            for topic_item in topic_links.get(paper.key, []):
                pair = (author_item["id"], topic_item["id"])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                author = next(item for item in kb.list_authors() if item.id == author_item["id"])
                topic = next(item for item in kb.list_topics() if item.id == topic_item["id"])
                path = synthesis_path(data_dir, author, topic)
                syntheses.append(
                    {
                        "author_id": author.id,
                        "author_label": format_author(author),
                        "topic_id": topic.id,
                        "topic_name": topic.name,
                        "has_synthesis": path.exists(),
                        "updated_at": file_updated(path),
                    }
                )

    paper_nodes.sort(key=lambda item: (-item["score"], item["title"].casefold()))
    return {
        "papers": paper_nodes,
        "authors": sorted(authors.values(), key=lambda item: item["name"].casefold()),
        "topics": sorted(topics.values(), key=lambda item: item["name"].casefold()),
        "edges": edges,
        "syntheses": syntheses,
    }
