from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from autopaper.analysis.biblio import extract_bibliography
from autopaper.errors import AutoPaperError, NotFoundError
from autopaper.models import Author, Paper, ScoredPaper, Topic
from autopaper.pipeline.dedup import same_paper
from autopaper.storage.db import KnowledgeBase
from autopaper.textutil import collapse_ws, normalize_title, slugify

PAPER_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass
class PlannedUpload:
    source: Path
    title: str
    folder: Path | None
    skip: bool
    reason: str
    paper_key: str = ""


def iter_upload_files(directory: Path, library_root: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        raise NotFoundError(f"找不到目录：{directory}")
    library_root = library_root.resolve()
    found: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in PAPER_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        resolved = path.resolve()
        if resolved == library_root or library_root in resolved.parents:
            continue
        found.append(path)
    found.sort(key=lambda item: str(item).casefold())
    return found


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", Path(name).name).strip().rstrip(".")
    return cleaned[:180] or "paper"


def folder_for_title(data_dir: Path, title: str) -> tuple[Path, bool]:
    """返回归类目录，以及这个论文名是否已经在本地存放。"""
    root = data_dir / "library"
    base = slugify(title)
    norm = normalize_title(title)
    direct = root / base
    if _folder_has_title(direct, norm):
        return direct, True
    digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    hashed = root / f"{base}-{digest}"
    if _folder_has_title(hashed, norm):
        return hashed, True
    if not (direct / "title.txt").exists():
        return direct, False
    return hashed, False


def _folder_has_title(folder: Path, title_norm: str) -> bool:
    marker = folder / "title.txt"
    if not marker.exists():
        return False
    stored = normalize_title(marker.read_text(encoding="utf-8"))
    return stored == title_norm


def plan_uploads(directory: Path, kb: KnowledgeBase, data_dir: Path) -> list[PlannedUpload]:
    files = iter_upload_files(directory, data_dir / "library")
    if not files:
        raise AutoPaperError("这个目录里没有 PDF、TXT 或 Markdown 论文。")
    seen: list[Paper] = []
    planned: list[PlannedUpload] = []
    for path in files:
        paper = _upload_identity(path)
        title = paper.title or path.name
        if not normalize_title(paper.title):
            planned.append(PlannedUpload(path, path.name, None, True, "文件名不能作为论文名"))
            continue
        if any(same_paper(paper, previous, 92) for previous in seen):
            planned.append(PlannedUpload(path, title, None, True, "本批已有同一篇论文"))
            continue
        seen.append(paper)
        match = kb.find_same_paper(paper)
        if match is not None:
            planned.append(PlannedUpload(path, title, None, True, f"知识库里已有同一篇：《{match.title}》"))
            continue
        folder, stored = folder_for_title(data_dir, title)
        if stored:
            planned.append(PlannedUpload(path, title, folder, True, "本地已按该论文名归类"))
            continue
        planned.append(PlannedUpload(path, title, folder, False, ""))
    return planned


def _upload_identity(path: Path) -> Paper:
    """用文件名做题名，并尽量读出 DOI。题名、副标题、DOI 和其他标识都能对上已有论文。"""
    paper = Paper(title=collapse_ws(path.stem), source="local", sources=["local"])
    try:
        found = extract_bibliography(path)
    except Exception:
        found = None
    if found is not None and found.doi:
        paper.doi = found.doi
    return paper.cleanup()


def store_uploads(
    planned: list[PlannedUpload],
    kb: KnowledgeBase,
    authors: list[Author],
    topics: list[Topic],
    by_name: dict[str, tuple[list[Author], list[Topic]]] | None = None,
) -> list[PlannedUpload]:
    stored: list[PlannedUpload] = []
    for item in planned:
        if item.skip or item.folder is None:
            stored.append(item)
            continue
        item_authors, item_topics = authors, topics
        if by_name is not None:
            chosen = by_name.get(item.source.name)
            if chosen is None:
                raise AutoPaperError(f"没有为《{item.title}》指定作者和主题")
            item_authors, item_topics = chosen
        folder = item.folder
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / safe_filename(item.source.name)
        shutil.copy2(item.source, target)
        (folder / "title.txt").write_text(item.title + "\n", encoding="utf-8")
        paper = Paper(
            title=item.title,
            authors=[author.name for author in item_authors],
            source="local",
            sources=["local"],
        )
        key = kb.upsert_paper(paper)
        kb.set_local_path(key, str(target.resolve()))
        for author in item_authors:
            kb.link_author_paper(author.id, key)
        for topic in item_topics:
            kb.link_topic_paper(
                topic.id,
                ScoredPaper(paper, 100, None, 100, "手动上传", "candidate"),
                key,
            )
        item.paper_key = key
        stored.append(item)
    return stored


def format_upload_report(
    items: list[PlannedUpload],
    authors: list[Author],
    topics: list[Topic],
) -> str:
    from autopaper.catalog import format_author

    uploaded = [item for item in items if not item.skip]
    skipped = [item for item in items if item.skip]
    lines = [f"上传 {len(uploaded)} 篇，跳过 {len(skipped)} 篇。"]
    if authors:
        lines.append("作者：" + "；".join(format_author(author) for author in authors))
    if topics:
        lines.append("主题：" + "；".join(topic.name for topic in topics))
    if uploaded:
        lines.append("已归类：")
        for item in uploaded:
            lines.append(f"  {item.title} -> {item.folder}")
    if skipped:
        lines.append("跳过：")
        for item in skipped:
            lines.append(f"  {item.source.name}（{item.reason}）")
    return "\n".join(lines)


def remove_paper_files(data_dir: Path, paper: Paper, kb: KnowledgeBase) -> bool:
    """删除这篇论文在论文库里的目录。目录外的文件不动，别的论文还在用的目录也不动。"""
    if not paper.local_path:
        return False
    folder = Path(paper.local_path).resolve().parent
    library = (data_dir / "library").resolve()
    if folder == library or library not in folder.parents:
        return False
    for other in kb.list_papers():
        if other.key == paper.key or not other.local_path:
            continue
        other_folder = Path(other.local_path).resolve().parent
        if other_folder == folder:
            return False
    shutil.rmtree(folder)
    return True


def remove_paper_note(paper: Paper) -> bool:
    if not paper.local_path:
        return False
    folder = Path(paper.local_path).resolve().parent
    removed = False
    for name in ("note.md", "note.json"):
        path = folder / name
        if path.exists() and path.is_file():
            path.unlink()
            removed = True
    return removed


def remove_author_note(data_dir: Path, author: Author) -> bool:
    from autopaper.analysis.notes import author_profile_path

    return _remove_file(author_profile_path(data_dir, author))


def remove_topic_note(data_dir: Path, topic: Topic) -> bool:
    from autopaper.analysis.notes import topic_profile_path

    return _remove_file(topic_profile_path(data_dir, topic))


def remove_synthesis_note(data_dir: Path, author: Author, topic: Topic) -> bool:
    from autopaper.analysis.notes import synthesis_path

    folder = synthesis_path(data_dir, author, topic).resolve().parent
    notes = (data_dir / "notes").resolve()
    if folder == notes or notes not in folder.parents:
        return False
    if not folder.exists():
        return False
    shutil.rmtree(folder)
    return True


def _remove_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    path.unlink()
    parent = path.parent
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
    return True
