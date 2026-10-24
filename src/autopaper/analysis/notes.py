from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from autopaper.analysis.reader import build_reading, load_document, model_label, render_reading
from autopaper.catalog import format_author
from autopaper.errors import AutoPaperError, LLMError
from autopaper.llm.prompts import (
    AUTHOR_PROFILE_SYSTEM,
    TOPIC_PROFILE_SYSTEM,
    SYNTHESIS_SYSTEM,
    author_profile_user_prompt,
    synthesis_user_prompt,
    topic_profile_user_prompt,
)
from autopaper.models import Author, Paper, Topic
from autopaper.storage.db import KnowledgeBase
from autopaper.textutil import slugify


def load_local_text(paper: Paper) -> tuple[str, bool, Path | None]:
    if paper.local_path:
        path = Path(paper.local_path)
        if path.exists():
            text = load_document(path).strip()
            if not text:
                raise AutoPaperError(f"文件里没有提取到文本：{path.name}")
            return text, False, path
    if paper.abstract.strip():
        return paper.abstract, True, None
    raise AutoPaperError(f"《{paper.title}》没有本地文件，也没有摘要。")


def write_paper_files(paper: Paper, payload: dict, markdown: str, model: str) -> Path | None:
    if not paper.local_path:
        return None
    folder = Path(paper.local_path).parent
    folder.mkdir(parents=True, exist_ok=True)
    note = folder / "note.md"
    note.write_text(markdown, encoding="utf-8")
    record = {"model": model_label(model) or model, "paper_key": paper.key, "reading": payload}
    note.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return note


def author_profile_path(data_dir: Path, author: Author) -> Path:
    return data_dir / "notes" / "authors" / (slugify(author.name) or "author") / "profile.md"


def topic_profile_path(data_dir: Path, topic: Topic) -> Path:
    return data_dir / "notes" / "topics" / (slugify(topic.name) or "topic") / "profile.md"


def file_updated(path: Path) -> str:
    if not path.exists():
        return ""
    moment = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0)
    return moment.isoformat()


def highlight_library_path(data_dir: Path) -> Path:
    return data_dir / "notes" / "highlights" / "library.md"


def write_highlight_library(data_dir: Path, rows: list[dict]) -> Path:
    path = highlight_library_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    titles = {"paper": "论文", "author": "作者", "topic": "主题", "synthesis": "综合笔记"}
    colors = {"yellow": "黄", "green": "绿", "blue": "蓝", "pink": "粉", "orange": "橙"}
    lines = [
        "# 我的笔记库",
        "",
        "这里保存论文、作者和主题笔记中的高亮与批注。更新笔记不会删除这些内容。",
        "",
    ]
    if not rows:
        lines.append("还没有高亮或批注。")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("scope") or ""), []).append(row)
    for scope in ("paper", "author", "topic", "synthesis"):
        items = grouped.get(scope) or []
        if not items:
            continue
        lines.extend([f"## {titles[scope]}", ""])
        owners: dict[str, list[dict]] = {}
        for item in items:
            owners.setdefault(str(item.get("scope_label") or item.get("scope_id") or "未命名"), []).append(item)
        for label, group in owners.items():
            lines.extend([f"### {label}", ""])
            for item in group:
                if item.get("kind") == "annotation":
                    kind = "批注"
                else:
                    kind = f"高亮（{colors.get(str(item.get('color') or ''), item.get('color') or '未标色')}）"
                lines.append(f"- {kind} · {item.get('updated_at') or ''}")
                lines.append(f"  - 段落：{item.get('paragraph') or ''}")
                selected = str(item.get("selected_text") or "")
                if selected and selected != item.get("paragraph"):
                    lines.append(f"  - 划取：{selected}")
                if item.get("comment"):
                    lines.append(f"  - 批注：{item['comment']}")
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def synthesis_path(data_dir: Path, author: Author, topic: Topic) -> Path:
    return _collection_dir(data_dir, [author], [topic]) / "synthesis.md"


def _collection_dir(data_dir: Path, authors: list[Author], topics: list[Topic]) -> Path:
    author_slug = slugify(" ".join(author.name for author in authors)) or "authors"
    topic_slug = slugify(" ".join(topic.name for topic in topics)) or "topics"
    return data_dir / "notes" / author_slug / topic_slug


def render_synthesis(authors: list[Author], topics: list[Topic], payload: dict, *, model: str) -> str:
    author_label = "；".join(format_author(author) for author in authors) or "未指定作者"
    topic_label = "；".join(topic.name for topic in topics) or "未指定主题"
    lines = [
        f"# {author_label}：{topic_label}",
        "",
        f"模型：{model_label(model) or model or '未记录'}",
        "",
        "这份笔记综合同一作者、同一主题下已经归类的论文，依据是各篇 `note.md`，不是在线作者主页。",
        "",
        "## 共同问题",
        "",
        str(payload.get("shared_problem") or "").strip() or "笔记未说明",
        "",
        "## 各篇角色",
        "",
    ]
    papers = payload.get("papers")
    if isinstance(papers, list) and papers:
        for item in papers:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip() or "未命名论文"
                role = str(item.get("role") or "").strip() or "笔记未说明"
                lines.append(f"- {title}：{role}")
            else:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
    else:
        lines.append("笔记未说明")
    lines.extend(
        [
            "",
            "## 方法线索",
            "",
            str(payload.get("method_thread") or "").strip() or "笔记未说明",
            "",
            "## 方向线索",
            "",
            str(payload.get("direction_thread") or "").strip() or "笔记未说明",
            "",
            "## 理论线索",
            "",
            str(payload.get("theory_thread") or "").strip() or "笔记未说明",
            "",
            "## 分歧",
            "",
            _bullet_lines(payload.get("tensions")),
            "",
            "## 仍未回答的问题",
            "",
            _bullet_lines(payload.get("open_questions")),
            "",
            "## 下一步",
            "",
            _bullet_lines(payload.get("next_steps")),
            "",
        ]
    )
    return "\n".join(lines)


def _bullet_lines(value) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text or "笔记未说明"
    if isinstance(value, list):
        rows = [f"- {str(item).strip()}" for item in value if str(item).strip()]
        return "\n".join(rows) if rows else "笔记未说明"
    return "笔记未说明"


def write_collection(
    data_dir: Path,
    authors: list[Author],
    topics: list[Topic],
    papers: list[tuple[Paper, Path | None]],
    synthesis_md: str,
    synthesis_payload: dict,
    model: str,
) -> Path:
    directory = _collection_dir(data_dir, authors, topics)
    directory.mkdir(parents=True, exist_ok=True)
    synthesis = directory / "synthesis.md"
    synthesis.write_text(synthesis_md, encoding="utf-8")
    synthesis.with_suffix(".json").write_text(
        json.dumps(
            {"model": model_label(model) or model, "synthesis": synthesis_payload},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    index_lines = [
        f"# { '；'.join(format_author(author) for author in authors) or '未指定作者' }",
        "",
        f"主题：{'；'.join(topic.name for topic in topics) or '未指定主题'}",
        "",
        f"- [综合笔记](synthesis.md)",
        "",
        "## 单篇笔记",
        "",
    ]
    for paper, note in papers:
        if note is not None:
            link = Path(os.path.relpath(note, directory)).as_posix()
            index_lines.append(f"- [{paper.title}]({link})")
        else:
            index_lines.append(f"- {paper.title}")
    index_lines.append("")
    (directory / "index.md").write_text("\n".join(index_lines), encoding="utf-8")
    return synthesis


def _note_excerpt(paper: Paper, payload: dict) -> str:
    kept = {
        "title": payload.get("title") or paper.title,
        "one_sentence": payload.get("one_sentence"),
        "field": payload.get("field"),
        "position": payload.get("position"),
        "problem": payload.get("problem"),
        "method": payload.get("method"),
        "results": payload.get("results"),
        "solves_problem": payload.get("solves_problem"),
        "innovations": payload.get("innovations"),
        "limitations": payload.get("limitations"),
        "new_questions": payload.get("new_questions"),
    }
    return json.dumps(kept, ensure_ascii=False)


def _model_name(llm) -> str:
    return getattr(getattr(llm, "settings", None), "llm_model", "")


def read_papers(kb: KnowledgeBase, llm, papers: list[Paper], *, echo=print) -> tuple[dict[str, dict], dict[str, Path | None], list[Path]]:
    """阅读给定论文并写入单篇笔记。已有综合笔记留给 refresh_syntheses。"""
    model = _model_name(llm)
    payloads: dict[str, dict] = {}
    note_paths: dict[str, Path | None] = {}
    written: list[Path] = []
    for index, paper in enumerate(papers, start=1):
        echo(f"[{index}/{len(papers)}] 阅读 {paper.title}")
        try:
            text, only_abstract, source = load_local_text(paper)
            payload, markdown = build_reading(
                paper,
                text,
                llm,
                only_abstract=only_abstract,
                source_name=source.name if source else "",
            )
        except (AutoPaperError, LLMError) as exc:
            echo(f"    跳过：{exc}")
            continue
        if not isinstance(payload, dict):
            payload = {}
        kb.save_reading(paper.key, payload, markdown)
        note = write_paper_files(paper, payload, markdown, model)
        payloads[paper.key] = payload
        note_paths[paper.key] = note
        if note is not None:
            written.append(note)
            echo(f"    笔记 {note}")
    return payloads, note_paths, written


def _stored_payload(kb: KnowledgeBase, paper: Paper) -> dict | None:
    stored = kb.get_reading(paper.key) or {}
    payload = stored.get("payload")
    if isinstance(payload, dict) and payload:
        return payload
    return None


def _note_path(paper: Paper, known: dict[str, Path | None]) -> Path | None:
    if paper.key in known and known[paper.key] is not None:
        return known[paper.key]
    if not paper.local_path:
        return None
    candidate = Path(paper.local_path).parent / "note.md"
    return candidate if candidate.exists() else None


def refresh_syntheses(
    kb: KnowledgeBase,
    data_dir: Path,
    llm,
    seed_papers: list[Paper],
    fresh: dict[str, dict],
    note_paths: dict[str, Path | None],
    *,
    echo=print,
) -> list[Path]:
    """用新笔记和已经保存的笔记，更新相关作者与主题的综合笔记。"""
    model = _model_name(llm)
    written: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for paper in seed_papers:
        authors = kb.authors_of_paper(paper.key)
        topics = kb.topics_of_paper(paper.key)
        pairs: list[tuple[list[Author], list[Topic], list[Paper]]] = []
        if authors and topics:
            for author in authors:
                for topic in topics:
                    pairs.append(([author], [topic], kb.list_local_papers(author_id=author.id, topic_id=topic.id)))
        else:
            pairs.append((authors, topics, [paper]))
        for author_list, topic_list, group in pairs:
            marker = (
                author_list[0].id if author_list else "",
                topic_list[0].id if topic_list else paper.key,
            )
            if marker in seen:
                continue
            seen.add(marker)
            rows = []
            for item in group:
                payload = fresh.get(item.key) or _stored_payload(kb, item)
                if payload is None:
                    continue
                rows.append((item, payload))
            if not rows or not any(item.key in fresh for item, _payload in rows):
                continue
            author_label = "；".join(format_author(item) for item in author_list) or "未指定作者"
            topic_label = "；".join(item.name for item in topic_list) or "未指定主题"
            echo(f"综合 {author_label} / {topic_label}")
            blocks = "\n\n".join(_note_excerpt(item, payload) for item, payload in rows)
            try:
                payload = llm.complete_json(
                    SYNTHESIS_SYSTEM,
                    synthesis_user_prompt(author_label, topic_label, blocks),
                    expect="object",
                    max_tokens=5000,
                )
            except (AutoPaperError, LLMError) as exc:
                echo(f"    综合笔记未更新：{exc}")
                continue
            if not isinstance(payload, dict):
                payload = {}
            markdown = render_synthesis(author_list, topic_list, payload, model=model)
            synthesis = write_collection(
                data_dir,
                author_list,
                topic_list,
                [(item, _note_path(item, note_paths)) for item, _payload in rows],
                markdown,
                payload,
                model,
            )
            written.append(synthesis)
            echo(f"    综合笔记 {synthesis}")
    return written


def refresh_pair(kb: KnowledgeBase, data_dir: Path, llm, author: Author, topic: Topic, *, echo=print) -> Path:
    """只根据已有单篇笔记，更新这一对作者和主题的综合笔记。"""
    rows = []
    for paper in kb.list_local_papers(author_id=author.id, topic_id=topic.id):
        payload = _stored_payload(kb, paper)
        if payload is not None:
            rows.append((paper, payload))
    if not rows:
        raise AutoPaperError("这些论文还没有单篇笔记。请先生成论文笔记。")
    model = _model_name(llm)
    author_label = format_author(author)
    topic_label = topic.name
    echo(f"综合 {author_label} / {topic_label}")
    blocks = "\n\n".join(_note_excerpt(item, payload) for item, payload in rows)
    payload = llm.complete_json(
        SYNTHESIS_SYSTEM,
        synthesis_user_prompt(author_label, topic_label, blocks),
        expect="object",
        max_tokens=5000,
    )
    if not isinstance(payload, dict):
        payload = {}
    markdown = render_synthesis([author], [topic], payload, model=model)
    return write_collection(
        data_dir,
        [author],
        [topic],
        [(item, _note_path(item, {})) for item, _payload in rows],
        markdown,
        payload,
        model,
    )


def note_papers(kb: KnowledgeBase, data_dir: Path, llm, papers: list[Paper], *, echo=print) -> list[Path]:
    """为指定论文生成笔记，并更新它们所在作者和主题的综合笔记。"""
    if not papers:
        return []
    payloads, note_paths, written = read_papers(kb, llm, papers, echo=echo)
    if not payloads:
        raise AutoPaperError("没有生成任何论文笔记。")
    written.extend(
        refresh_syntheses(kb, data_dir, llm, papers, payloads, note_paths, echo=echo)
    )
    return written


def run_notes(
    kb: KnowledgeBase,
    data_dir: Path,
    llm,
    *,
    author: Author | None = None,
    topic: Topic | None = None,
    missing_only: bool = False,
    echo=print,
) -> list[Path]:
    papers = kb.list_local_papers(
        author_id=author.id if author else None,
        topic_id=topic.id if topic else None,
    )
    if missing_only:
        papers = [paper for paper in papers if _stored_payload(kb, paper) is None]
    if not papers:
        raise AutoPaperError("没有找到需要生成笔记的本地论文。")
    return note_papers(kb, data_dir, llm, papers, echo=echo)


def _profile_bundle(kb: KnowledgeBase, papers: list[Paper]) -> tuple[str, str]:
    blocks = []
    missing = []
    for paper in papers:
        payload = _stored_payload(kb, paper)
        if payload is None:
            missing.append(paper.title)
            continue
        authors = "；".join(format_author(author) for author in kb.authors_of_paper(paper.key)) or "未指定作者"
        topics = "；".join(topic.name for topic in kb.topics_of_paper(paper.key)) or "未指定主题"
        blocks.append(f"归类作者：{authors}\n归类主题：{topics}\n{_note_excerpt(paper, payload)}")
    if not blocks:
        raise AutoPaperError("这些论文还没有单篇笔记。请先上传或运行 autopaper notes。")
    return "\n\n".join(blocks), "；".join(missing)


def _write_profile(path: Path, markdown: str, payload: dict, model: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps({"model": model_label(model) or model, "profile": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _scholars(value) -> str:
    if not isinstance(value, list) or not value:
        return "笔记未说明"
    rows = []
    for item in value:
        if not isinstance(item, dict):
            text = str(item).strip()
            if text:
                rows.append(f"- {text}")
            continue
        name = str(item.get("name") or "").strip() or "未点名"
        relation = str(item.get("relation") or "").strip() or "笔记未说明"
        evidence = str(item.get("evidence") or "").strip() or "笔记未说明"
        rows.append(f"- {name}：{relation}。依据：{evidence}")
    return "\n".join(rows) if rows else "笔记未说明"


def _topic_relations(value) -> str:
    if not isinstance(value, list) or not value:
        return "笔记未说明"
    rows = []
    for item in value:
        if not isinstance(item, dict):
            text = str(item).strip()
            if text:
                rows.append(f"- {text}")
            continue
        topic = str(item.get("topic") or "").strip() or "未命名主题"
        relation = str(item.get("relation") or "").strip() or "笔记未说明"
        rows.append(f"- {topic}：{relation}")
    return "\n".join(rows) if rows else "笔记未说明"


def render_author_profile(author: Author, payload: dict, *, model: str) -> str:
    relations = payload.get("relationships") if isinstance(payload.get("relationships"), dict) else {}
    works = payload.get("works")
    work_lines = []
    if isinstance(works, list):
        for item in works:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip() or "未命名论文"
                role = str(item.get("role") or "").strip() or "笔记未说明"
                work_lines.append(f"- {title}：{role}")
    lines = [
        f"# {format_author(author)}",
        "",
        f"模型：{model_label(model) or model or '未记录'}",
        "",
        "这份笔记只依据该作者已经归类、并且已经生成单篇笔记的论文，不是在线作者主页。",
        "",
        "## 研究方向",
        "",
        str(payload.get("direction") or "").strip() or "笔记未说明",
        "",
        "## 定位",
        "",
        str(payload.get("positioning") or "").strip() or "笔记未说明",
        "",
        "## 学术关系",
        "",
        "### 论文之间",
        "",
        str(relations.get("among_papers") or "").strip() or "笔记未说明",
        "",
        "### 学者",
        "",
        _scholars(relations.get("scholars")),
        "",
        "### 主题",
        "",
        _topic_relations(relations.get("topics")),
        "",
        "## 各篇角色",
        "",
        "\n".join(work_lines) if work_lines else "笔记未说明",
        "",
        "## 仍未回答的问题",
        "",
        _bullet_lines(payload.get("open_questions")),
        "",
        "## 这批笔记覆盖不到的地方",
        "",
        str(payload.get("evidence_limits") or "").strip() or "笔记未说明",
        "",
    ]
    return "\n".join(lines)


def render_topic_profile(topic: Topic, payload: dict, *, model: str) -> str:
    authors = payload.get("authors")
    author_blocks = []
    if isinstance(authors, list):
        for item in authors:
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    author_blocks.append(text)
                continue
            name = str(item.get("name") or "").strip() or "未命名作者"
            author_blocks.extend(
                [
                    f"### {name}",
                    "",
                    f"研究方向：{str(item.get('direction') or '').strip() or '笔记未说明'}",
                    "",
                    f"定位：{str(item.get('positioning') or '').strip() or '笔记未说明'}",
                    "",
                    f"学术关系：{str(item.get('relationships') or '').strip() or '笔记未说明'}",
                    "",
                ]
            )
    relations = payload.get("paper_relations")
    relation_lines = []
    if isinstance(relations, list):
        for item in relations:
            if isinstance(item, dict):
                papers = str(item.get("papers") or "").strip() or "未命名论文"
                relation = str(item.get("relation") or "").strip() or "笔记未说明"
                relation_lines.append(f"- {papers}：{relation}")
    lines = [
        f"# {topic.name}",
        "",
        f"模型：{model_label(model) or model or '未记录'}",
        "",
        "这份笔记只依据该主题下已经生成单篇笔记的论文。",
        "",
        "## 主题概况",
        "",
        str(payload.get("overview") or "").strip() or "笔记未说明",
        "",
        "## 作者",
        "",
        "\n".join(author_blocks).strip() or "笔记未说明",
        "",
        "## 论文关系",
        "",
        "\n".join(relation_lines) if relation_lines else "笔记未说明",
        "",
        "## 分歧",
        "",
        _bullet_lines(payload.get("tensions")),
        "",
        "## 仍未回答的问题",
        "",
        _bullet_lines(payload.get("open_questions")),
        "",
    ]
    return "\n".join(lines)


def generate_author_profile(kb: KnowledgeBase, data_dir: Path, llm, author: Author) -> Path:
    papers = kb.list_local_papers(author_id=author.id)
    if not papers:
        raise AutoPaperError(f"{format_author(author)} 还没有已归类的论文。")
    notes, missing = _profile_bundle(kb, papers)
    payload = llm.complete_json(
        AUTHOR_PROFILE_SYSTEM,
        author_profile_user_prompt(
            format_author(author),
            " / ".join(author.aliases),
            missing,
            notes,
        ),
        expect="object",
        max_tokens=5000,
    )
    if not isinstance(payload, dict):
        payload = {}
    model = _model_name(llm)
    markdown = render_author_profile(author, payload, model=model)
    return _write_profile(author_profile_path(data_dir, author), markdown, payload, model)


def generate_topic_profile(kb: KnowledgeBase, data_dir: Path, llm, topic: Topic) -> Path:
    papers = kb.list_local_papers(topic_id=topic.id)
    if not papers:
        raise AutoPaperError(f"主题「{topic.name}」还没有已归类的论文。")
    notes, missing = _profile_bundle(kb, papers)
    payload = llm.complete_json(
        TOPIC_PROFILE_SYSTEM,
        topic_profile_user_prompt(topic.name, missing, notes),
        expect="object",
        max_tokens=5000,
    )
    if not isinstance(payload, dict):
        payload = {}
    model = _model_name(llm)
    markdown = render_topic_profile(topic, payload, model=model)
    return _write_profile(topic_profile_path(data_dir, topic), markdown, payload, model)
