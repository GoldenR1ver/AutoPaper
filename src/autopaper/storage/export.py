from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

from autopaper.models import Paper, ScoredPaper


def format_reference(index: int, paper: Paper) -> str:
    authors = ", ".join(paper.authors[:6])
    if len(paper.authors) > 6:
        authors += ", et al."
    if not authors:
        authors = "Unknown"
    parts = [f"{index}. {authors}. {paper.title}."]
    if paper.venue:
        parts.append(f" {paper.venue}.")
    if paper.year:
        parts.append(f" {paper.year}.")
    if paper.doi:
        parts.append(f" https://doi.org/{paper.doi}")
    elif paper.url:
        parts.append(f" {paper.url}")
    return "".join(parts)


def reference_list(papers: list[Paper]) -> str:
    if not papers:
        return "（没有论文）"
    return "\n".join(format_reference(index, paper) for index, paper in enumerate(papers, start=1))


def _bibtex_escape(value: str) -> str:
    value = value.replace("\\", r"\textbackslash{}")
    for char, replacement in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        value = value.replace(char, replacement)
    return value


def _bibtex_key(paper: Paper, used: set[str]) -> str:
    author = "unknown"
    if paper.authors:
        last = re.sub(r"[^\w]+", "", paper.authors[0].split()[-1], flags=re.UNICODE).casefold()
        author = last or "unknown"
    year = str(paper.year or "nd")
    word = ""
    stop = {"with", "from", "that", "this", "using", "into", "over"}
    for token in re.split(r"\W+", paper.title):
        if len(token) > 3 and token.casefold() not in stop:
            word = re.sub(r"[^\w]+", "", token, flags=re.UNICODE).casefold()
            break
    base = f"{author}{year}{word}"[:60] or "paper"
    key = base
    suffix = 2
    while key in used:
        key = f"{base}{suffix}"
        suffix += 1
    used.add(key)
    return key


def render_bibtex(papers: list[Paper]) -> str:
    used: set[str] = set()
    entries = []
    for paper in papers:
        key = _bibtex_key(paper, used)
        entry_type = "article" if paper.venue else "misc"
        fields = [f"  title = {{{_bibtex_escape(paper.title)}}}"]
        if paper.authors:
            fields.append("  author = {" + " and ".join(_bibtex_escape(name) for name in paper.authors) + "}")
        if paper.year:
            fields.append(f"  year = {{{paper.year}}}")
        if paper.venue:
            fields.append(f"  journal = {{{_bibtex_escape(paper.venue)}}}")
        elif paper.arxiv_id:
            fields.append(f"  journal = {{arXiv preprint arXiv:{paper.arxiv_id}}}")
        if paper.doi:
            fields.append(f"  doi = {{{_bibtex_escape(paper.doi)}}}")
        url = paper.url or (f"https://doi.org/{paper.doi}" if paper.doi else None)
        if url:
            fields.append(f"  url = {{{_bibtex_escape(url)}}}")
        entries.append(f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}")
    return "\n".join(entries) + ("\n" if entries else "")


def write_bibtex(path: Path, papers: list[Paper]) -> None:
    path.write_text(render_bibtex(papers), encoding="utf-8")


def _row(item: ScoredPaper) -> dict:
    paper = item.paper
    return {
        "key": paper.key,
        "title": paper.title,
        "authors": "; ".join(paper.authors),
        "year": paper.year or "",
        "venue": paper.venue or "",
        "doi": paper.doi or "",
        "url": paper.url or "",
        "pdf_url": paper.pdf_url or "",
        "citation_count": paper.citation_count if paper.citation_count is not None else "",
        "score": item.score,
        "heuristic_score": item.heuristic_score,
        "llm_score": "" if item.llm_score is None else item.llm_score,
        "status": item.status,
        "reason": item.reason,
        "categories": "; ".join(paper.categories),
        "sources": "; ".join(paper.sources),
        "abstract": paper.abstract,
    }


def write_csv(path: Path, items: list[ScoredPaper]) -> None:
    rows = [_row(item) for item in items]
    fieldnames = list(rows[0]) if rows else list(_row(ScoredPaper(Paper(title=""), 0, None, 0, "", "candidate")))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, items: list[ScoredPaper]) -> None:
    import json

    path.write_text(json.dumps([_row(item) for item in items], ensure_ascii=False, indent=2), encoding="utf-8")


def render_candidate_markdown(topic_name: str, items: list[ScoredPaper]) -> str:
    lines = [f"# {topic_name} 候选论文", "", f"共 {len(items)} 篇。", ""]
    if not items:
        lines.append("这次没有候选论文。可以放宽关键词、领域标签或最低分。")
        return "\n".join(lines) + "\n"
    lines.extend(["| 分数 | 年份 | 被引 | 标题 |", "| ---: | ---: | ---: | --- |"])
    for item in items:
        paper = item.paper
        title = paper.title.replace("|", " ")
        lines.append(f"| {item.score:.1f} | {paper.year or ''} | {paper.citation_count or 0} | {title} |")
    lines.extend(["", "## 摘要", ""])
    for item in items:
        paper = item.paper
        lines.append(f"### {paper.title}")
        lines.append("")
        lines.append(f"- 分数：{item.score:.1f}（{item.reason}）")
        if paper.authors:
            lines.append(f"- 作者：{', '.join(paper.authors)}")
        if paper.venue or paper.year:
            lines.append(f"- 出处：{paper.venue or '未知'} {paper.year or ''}".rstrip())
        if paper.doi:
            lines.append(f"- DOI：https://doi.org/{paper.doi}")
        elif paper.url:
            lines.append(f"- 链接：{paper.url}")
        abstract = paper.abstract[:500] + ("…" if len(paper.abstract) > 500 else "")
        lines.append("")
        lines.append(abstract or "（无摘要）")
        lines.append("")
    lines.extend(["## 引用列表", "", reference_list([item.paper for item in items]), ""])
    return "\n".join(lines)


def export_topic(dest: Path, topic_name: str, items: list[ScoredPaper], formats: set[str]) -> dict[str, Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    papers = [item.paper for item in items]
    if "csv" in formats:
        path = dest / "papers.csv"
        write_csv(path, items)
        written["csv"] = path
    if "json" in formats:
        path = dest / "papers.json"
        write_json(path, items)
        written["json"] = path
    if "bib" in formats or "bibtex" in formats:
        path = dest / "papers.bib"
        write_bibtex(path, papers)
        written["bib"] = path
    if "md" in formats or "markdown" in formats:
        path = dest / "candidates.md"
        path.write_text(render_candidate_markdown(topic_name, items), encoding="utf-8")
        written["md"] = path
    return written


def export_topic_bundle(data_dir: Path, topic_id: str, topic_name: str, items: list[ScoredPaper]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dated = data_dir / "exports" / topic_id / stamp
    latest = data_dir / "exports" / topic_id / "latest"
    export_topic(dated, topic_name, items, {"csv", "json", "bib", "md"})
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(dated, latest)
    return latest
