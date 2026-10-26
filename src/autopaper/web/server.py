from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from autopaper.analysis.biblio import fill_local_papers, fill_paper, format_recognized, recognize_uploaded, set_bibliography
from autopaper.analysis.classify import reclassify_library, suggest_many
from autopaper.analysis.excerpt import build_excerpt
from autopaper.analysis.reader import load_catalog_source
from autopaper.analysis.pdfnotes import annotate_pdf
from autopaper.analysis.notes import (
    author_profile_path,
    file_updated,
    generate_author_profile,
    generate_topic_profile,
    note_papers,
    refresh_pair,
    synthesis_path,
    topic_profile_path,
    write_highlight_library,
)
from autopaper.llm.client import LLMClient
from autopaper.llm.prompts import FOLLOWUP_SYSTEM, followup_question, followup_user_prompt
from autopaper.catalog import format_author, require_existing_author, resolve_authors, resolve_topics
from autopaper.config import Settings
from autopaper.errors import AutoPaperError, NotFoundError
from autopaper.library import (
    plan_uploads,
    remove_author_note,
    remove_paper_files,
    remove_paper_note,
    remove_synthesis_note,
    remove_topic_note,
    safe_filename,
    store_uploads,
)
from autopaper.storage.db import KnowledgeBase
from autopaper.web.graph import build_library_graph

STATIC_DIR = Path(__file__).resolve().parent / "static"
HIGHLIGHT_COLORS = {"yellow", "green", "blue", "pink", "orange"}
HIGHLIGHT_SCOPES = {"paper", "author", "topic", "synthesis"}


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, list[str]], list[tuple[str, bytes]]]:
    match = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type or "")
    if not match:
        raise AutoPaperError("上传内容不是表单")
    boundary = (match.group(1) or match.group(2)).encode("utf-8", "replace")
    fields: dict[str, list[str]] = {}
    files: list[tuple[str, bytes]] = []
    for part in body.split(b"--" + boundary):
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2]
            if part.endswith(b"\r\n"):
                part = part[:-2]
        header_blob, separator, data = part.partition(b"\r\n\r\n")
        if separator == b"":
            continue
        headers: dict[str, str] = {}
        for line in header_blob.decode("utf-8", "replace").split("\r\n"):
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
        disposition = headers.get("content-disposition", "")
        field_name = _disposition_value(disposition, "name")
        filename = _disposition_value(disposition, "filename")
        if not field_name:
            continue
        if filename:
            files.append((Path(filename).name, data))
        else:
            fields.setdefault(field_name, []).append(data.decode("utf-8", "replace"))
    return fields, files


def _disposition_value(header: str, key: str) -> str:
    match = re.search(rf'{key}="([^"]*)"', header)
    if match:
        return match.group(1)
    match = re.search(rf"{key}=([^;]+)", header)
    return match.group(1).strip() if match else ""


class LibraryApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()

    def graph(self) -> dict:
        with self.lock, self._kb() as kb:
            return build_library_graph(kb, self.settings.data_dir)

    def paper(self, key: str) -> dict:
        with self.lock, self._kb() as kb:
            paper = kb.resolve_paper(key)
            if paper is None:
                raise NotFoundError(f"没有找到论文：{key}")
            graph = build_library_graph(kb, self.settings.data_dir)
            node = next((item for item in graph["papers"] if item["key"] == paper.key), None)
            if node is None:
                raise NotFoundError(f"没有找到论文：{key}")
            reading = kb.get_reading(paper.key) or {}
            related = []
            for edge in graph["edges"]:
                other = edge["target"] if edge["source"] == paper.key else edge["source"] if edge["target"] == paper.key else ""
                if not other:
                    continue
                match = next((item for item in graph["papers"] if item["key"] == other), None)
                if match is None:
                    continue
                related.append(
                    {
                        "key": other,
                        "title": match["title"],
                        "weight": edge["weight"],
                        "reason": edge["reason"],
                        "has_note": match["has_note"],
                    }
                )
            related.sort(key=lambda item: (-item["weight"], item["title"].casefold()))
            return {
                "paper": node,
                "reading": reading.get("payload") or {},
                "markdown": reading.get("markdown") or "",
                "updated_at": reading.get("created_at") or "",
                "related": related[:12],
            }

    def file_path(self, key: str) -> Path:
        with self.lock, self._kb() as kb:
            paper = kb.resolve_paper(key)
            if paper is None or not paper.local_path:
                raise NotFoundError("这篇论文没有本地文件")
            path = Path(paper.local_path).resolve()
            library = (self.settings.data_dir / "library").resolve()
            if library not in path.parents:
                raise NotFoundError("本地文件不在论文库中")
            if not path.exists():
                raise NotFoundError("本地文件已经不在原位置")
            return path

    def open_local(self, key: str) -> dict:
        path = self.file_path(key)
        try:
            launch_local_file(path)
        except OSError as exc:
            raise AutoPaperError(f"无法用本机阅读器打开：{exc}") from exc
        return {"ok": True}

    def annotate_original(self, key: str) -> dict:
        path = self.file_path(key)
        if path.suffix.lower() != ".pdf":
            raise AutoPaperError("自动批注目前只写入 PDF。")
        llm = _open_llm(self.settings)
        if llm is None:
            raise AutoPaperError("自动批注需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。")
        try:
            return annotate_pdf(path, llm).to_dict()
        finally:
            llm.close()

    def synthesis(self, author_id: str, topic_id: str) -> dict:
        with self.lock, self._kb() as kb:
            author = require_existing_author(kb, author_id)
            topic = kb.get_topic(topic_id)
            path = synthesis_path(self.settings.data_dir, author, topic)
            if not path.exists():
                raise NotFoundError("还没有这组论文的综合笔记。可以先运行 autopaper notes。")
            return {"markdown": path.read_text(encoding="utf-8"), "updated_at": file_updated(path)}

    def update_synthesis(self, author_id: str, topic_id: str) -> dict:
        llm = _open_llm(self.settings)
        if llm is None:
            raise AutoPaperError("生成笔记需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。")
        try:
            with self._kb() as kb:
                author = require_existing_author(kb, author_id)
                topic = kb.get_topic(topic_id)
                path = refresh_pair(kb, self.settings.data_dir, llm, author, topic)
            return {"markdown": path.read_text(encoding="utf-8"), "updated_at": file_updated(path)}
        finally:
            llm.close()

    def update_paper_note(self, key: str) -> dict:
        llm = _open_llm(self.settings)
        if llm is None:
            raise AutoPaperError("生成笔记需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。")
        try:
            with self._kb() as kb:
                paper = kb.resolve_paper(key)
                if paper is None:
                    raise NotFoundError(f"没有找到论文：{key}")
                note_papers(kb, self.settings.data_dir, llm, [paper])
                reading = kb.get_reading(paper.key) or {}
            return {"markdown": reading.get("markdown") or "", "updated_at": reading.get("created_at") or ""}
        finally:
            llm.close()

    def read_profile(self, kind: str, ident: str) -> dict:
        with self.lock, self._kb() as kb:
            path = self._profile_path(kb, kind, ident)
            if not path.exists():
                raise NotFoundError("还没有这份笔记。")
            return {"markdown": path.read_text(encoding="utf-8"), "updated_at": file_updated(path)}

    def write_profile(self, kind: str, ident: str) -> dict:
        llm = _open_llm(self.settings)
        if llm is None:
            raise AutoPaperError("生成笔记需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。")
        try:
            with self._kb() as kb:
                if kind == "author":
                    author = require_existing_author(kb, ident)
                    path = generate_author_profile(kb, self.settings.data_dir, llm, author)
                elif kind == "topic":
                    topic = kb.get_topic(ident)
                    path = generate_topic_profile(kb, self.settings.data_dir, llm, topic)
                else:
                    raise AutoPaperError("kind 必须是 author 或 topic")
            return {"markdown": path.read_text(encoding="utf-8"), "path": str(path), "updated_at": file_updated(path)}
        finally:
            llm.close()

    def _profile_path(self, kb: KnowledgeBase, kind: str, ident: str) -> Path:
        if kind == "author":
            return author_profile_path(self.settings.data_dir, require_existing_author(kb, ident))
        if kind == "topic":
            return topic_profile_path(self.settings.data_dir, kb.get_topic(ident))
        raise AutoPaperError("kind 必须是 author 或 topic")

    def highlights(self, scope: str = "", scope_id: str = "") -> list[dict]:
        with self.lock, self._kb() as kb:
            return kb.list_highlights(scope, scope_id)

    def save_highlights(self, payload: dict) -> dict:
        scope = str(payload.get("scope") or "")
        scope_id = str(payload.get("scope_id") or "").strip()
        scope_label = str(payload.get("scope_label") or "").strip()
        kind = str(payload.get("kind") or "")
        color = str(payload.get("color") or "")
        comment = str(payload.get("comment") or "").strip()
        if scope not in HIGHLIGHT_SCOPES:
            raise AutoPaperError("高亮范围必须是论文、作者、主题或综合笔记")
        if not scope_id:
            raise AutoPaperError("缺少笔记对象")
        if kind not in {"highlight", "annotation"}:
            raise AutoPaperError("请选择高亮或批注")
        if kind == "highlight":
            if color not in HIGHLIGHT_COLORS:
                raise AutoPaperError("请选择高亮颜色")
            comment = ""
        else:
            if color not in HIGHLIGHT_COLORS:
                color = ""
            if not comment:
                raise AutoPaperError("请写下批注")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raw_items = [{"paragraph": payload.get("paragraph"), "selected_text": payload.get("selected_text")}]
        cleaned = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            paragraph = " ".join(str(item.get("paragraph") or "").split())
            selected = " ".join(str(item.get("selected_text") or "").split())
            if not paragraph:
                continue
            cleaned.append((paragraph[:8000], (selected or paragraph)[:4000]))
        if not cleaned:
            raise AutoPaperError("没有可保存的段落")
        with self.lock, self._kb() as kb:
            self._require_scope(kb, scope, scope_id)
            saved = [
                kb.add_highlight(
                    scope=scope,
                    scope_id=scope_id,
                    scope_label=scope_label[:300],
                    kind=kind,
                    color=color,
                    paragraph=paragraph,
                    selected_text=selected,
                    comment=comment[:4000],
                )
                for paragraph, selected in cleaned
            ]
            rows = kb.list_highlights()
        write_highlight_library(self.settings.data_dir, rows)
        return {"highlights": saved}

    def ask(self, payload: dict) -> dict:
        question_id = str(payload.get("question") or "")
        custom = " ".join(str(payload.get("custom") or "").split())
        if question_id == "custom":
            if not custom:
                raise AutoPaperError("请写下要追问的问题")
            label = custom[:40]
            instruction = f"按这个问题回答，仍然只依据给定笔记：{custom[:200]}"
        else:
            chosen = followup_question(question_id)
            if chosen is None:
                raise AutoPaperError("没有这个问题")
            label, instruction = chosen
        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        paragraphs = []
        selected = []
        for item in items:
            if not isinstance(item, dict):
                continue
            paragraph = " ".join(str(item.get("paragraph") or "").split())
            piece = " ".join(str(item.get("selected_text") or "").split())
            if paragraph:
                paragraphs.append(paragraph[:4000])
            if piece:
                selected.append(piece[:2000])
        if not paragraphs and not selected:
            raise AutoPaperError("请先划取笔记中的段落")
        scope = str(payload.get("scope") or "")
        scope_id = str(payload.get("scope_id") or "").strip()
        context = self._ask_context(scope, scope_id)
        note = str(payload.get("note") or "").strip() or context["note"]
        llm = _open_llm(self.settings)
        if llm is None:
            raise AutoPaperError("追问需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。")
        try:
            result = llm.complete_json(
                FOLLOWUP_SYSTEM,
                followup_user_prompt(
                    scope_name=context["scope_name"],
                    label=str(payload.get("scope_label") or "").strip() or context["label"],
                    authors=context["authors"],
                    topics=context["topics"],
                    instruction=instruction,
                    selected="\n".join(selected),
                    paragraphs="\n\n".join(paragraphs),
                    note=note,
                ),
                expect="object",
                max_tokens=500,
            )
        finally:
            llm.close()
        answer = ""
        if isinstance(result, dict):
            answer = str(result.get("answer") or "").strip()
        if not answer:
            raise AutoPaperError("模型没有给出回答")
        return {"answer": answer, "question": label}

    def _ask_context(self, scope: str, scope_id: str) -> dict:
        with self.lock, self._kb() as kb:
            self._require_scope(kb, scope, scope_id)
            if scope == "paper":
                paper = kb.resolve_paper(scope_id)
                reading = kb.get_reading(paper.key) or {}
                authors = "；".join(format_author(author) for author in kb.authors_of_paper(paper.key))
                topics = "；".join(topic.name for topic in kb.topics_of_paper(paper.key))
                return {
                    "scope_name": "论文笔记",
                    "label": paper.title,
                    "authors": authors,
                    "topics": topics,
                    "note": reading.get("markdown") or "",
                }
            if scope == "author":
                author = require_existing_author(kb, scope_id)
                path = author_profile_path(self.settings.data_dir, author)
                return {
                    "scope_name": "作者笔记",
                    "label": format_author(author),
                    "authors": format_author(author),
                    "topics": "",
                    "note": path.read_text(encoding="utf-8") if path.exists() else "",
                }
            if scope == "topic":
                topic = kb.get_topic(scope_id)
                path = topic_profile_path(self.settings.data_dir, topic)
                return {
                    "scope_name": "主题笔记",
                    "label": topic.name,
                    "authors": "",
                    "topics": topic.name,
                    "note": path.read_text(encoding="utf-8") if path.exists() else "",
                }
            author_id, _, topic_id = scope_id.partition("|")
            author = require_existing_author(kb, author_id)
            topic = kb.get_topic(topic_id)
            path = synthesis_path(self.settings.data_dir, author, topic)
            return {
                "scope_name": "综合笔记",
                "label": f"{format_author(author)} / {topic.name}",
                "authors": format_author(author),
                "topics": topic.name,
                "note": path.read_text(encoding="utf-8") if path.exists() else "",
            }

    def remove_highlight(self, ident: str) -> dict:
        with self.lock, self._kb() as kb:
            kb.delete_highlight(ident)
            rows = kb.list_highlights()
        write_highlight_library(self.settings.data_dir, rows)
        return {"ok": True}

    def duplicate_papers(self) -> dict:
        with self.lock, self._kb() as kb:
            groups = kb.duplicate_groups()
        return {
            "groups": [
                [{"key": paper.key, "title": paper.title, "doi": paper.doi or ""} for paper in group]
                for group in groups
            ],
            "message": _duplicate_message(groups),
        }

    def merge_duplicates(self) -> dict:
        with self.lock, self._kb() as kb:
            merged = kb.merge_same_papers()
            rows = kb.list_highlights()
        write_highlight_library(self.settings.data_dir, rows)
        if not merged:
            return {"ok": True, "merged": 0, "groups": [], "message": "没有发现重复论文。"}
        lines = []
        for item in merged:
            joined = "、".join(f"《{title}》" for title in item["removed"])
            line = f"已把 {joined} 并入《{item['title']}》。"
            if item["extra_files"]:
                line += "另一份原文还在磁盘上，可以稍后自行删除。"
            lines.append(line)
        return {"ok": True, "merged": len(merged), "groups": merged, "message": "".join(lines)}

    def remove_paper(self, key: str) -> dict:
        with self.lock, self._kb() as kb:
            paper = kb.resolve_paper(key)
            if paper is None:
                raise NotFoundError(f"没有找到论文：{key}")
            remove_paper_files(self.settings.data_dir, paper, kb)
            paper = kb.delete_paper(paper.key)
            rows = kb.list_highlights()
        write_highlight_library(self.settings.data_dir, rows)
        return {"ok": True, "message": f"已删除论文《{paper.title}》。"}

    def remove_author(self, author_id: str) -> dict:
        with self.lock, self._kb() as kb:
            author = kb.delete_author(author_id)
            remove_author_note(self.settings.data_dir, author)
            rows = kb.list_highlights()
        write_highlight_library(self.settings.data_dir, rows)
        return {"ok": True, "message": f"已删除作者 {format_author(author)}。论文仍留在库里。"}

    def remove_topic(self, topic_id: str) -> dict:
        with self.lock, self._kb() as kb:
            topic = kb.delete_topic_record(topic_id)
            remove_topic_note(self.settings.data_dir, topic)
            rows = kb.list_highlights()
        write_highlight_library(self.settings.data_dir, rows)
        return {"ok": True, "message": f"已删除主题「{topic.name}」。论文仍留在库里。"}

    def remove_note(self, scope: str, scope_id: str, author_id: str = "", topic_id: str = "") -> dict:
        with self.lock, self._kb() as kb:
            if scope == "paper":
                paper = kb.delete_reading(scope_id)
                remove_paper_note(paper)
                return {"ok": True, "message": f"已删除《{paper.title}》的笔记。论文和原文还在。"}
            if scope == "author":
                author = kb.get_author(scope_id)
                remove_author_note(self.settings.data_dir, author)
                return {"ok": True, "message": f"已删除 {format_author(author)} 的作者笔记。"}
            if scope == "topic":
                topic = kb.get_topic(scope_id)
                remove_topic_note(self.settings.data_dir, topic)
                return {"ok": True, "message": f"已删除主题「{topic.name}」的笔记。"}
            if scope == "synthesis":
                author = require_existing_author(kb, author_id)
                topic = kb.get_topic(topic_id)
                remove_synthesis_note(self.settings.data_dir, author, topic)
                return {"ok": True, "message": "已删除这组综合笔记。"}
            raise AutoPaperError("scope 必须是 paper、author、topic 或 synthesis")

    def _require_scope(self, kb: KnowledgeBase, scope: str, scope_id: str) -> None:
        if scope == "paper":
            if kb.resolve_paper(scope_id) is None:
                raise NotFoundError("没有找到这篇论文")
            return
        if scope == "author":
            require_existing_author(kb, scope_id)
            return
        if scope == "topic":
            kb.get_topic(scope_id)
            return
        author_id, separator, topic_id = scope_id.partition("|")
        if not separator or not author_id or not topic_id:
            raise AutoPaperError("综合笔记缺少作者或主题")
        require_existing_author(kb, author_id)
        kb.get_topic(topic_id)

    def update_alias(self, author_ref: str, add: list[str], remove: list[str]) -> dict:
        with self.lock, self._kb() as kb:
            author = require_existing_author(kb, author_ref)
            author = kb.update_aliases(author.id, add=add, remove=remove)
            from autopaper.catalog import format_author

            return {"id": author.id, "label": format_author(author), "aliases": author.aliases}

    def fill_bibliography(self, key: str = "", *, all_papers: bool = False) -> dict:
        with self.lock, self._kb() as kb:
            if all_papers:
                records = fill_local_papers(kb)
            else:
                paper = kb.resolve_paper(key)
                if paper is None:
                    raise NotFoundError(f"没有找到论文：{key}")
                if not paper.local_path:
                    raise AutoPaperError("这篇论文没有本地原文，无法从原文补全。")
                records = [fill_paper(kb, paper, only_missing=True)]
        return {
            "papers": records,
            "count": len(records),
            "filled_count": sum(1 for item in records if item["filled"]),
        }

    def save_bibliography(self, key: str, year_raw, venue: str, doi: str, url: str) -> dict:
        text = str(year_raw or "").strip()
        year = None
        if text:
            if not re.fullmatch(r"\d{4}", text):
                raise AutoPaperError("年份请填四位数字，或留空。")
            year = int(text)
            if not 1900 <= year <= 2030:
                raise AutoPaperError("年份请填 1900 到 2030 之间的数字，或留空。")
        with self.lock, self._kb() as kb:
            paper = kb.resolve_paper(key)
            if paper is None:
                raise NotFoundError(f"没有找到论文：{key}")
            return set_bibliography(kb, paper, year=year, venue=venue, doi=doi, url=url)

    def suggest_upload(self, files: list[tuple[str, bytes]]) -> dict:
        if not files:
            raise AutoPaperError("请先选择论文文件。")
        excerpts = []
        with tempfile.TemporaryDirectory(prefix="autopaper-suggest-") as temporary:
            root = Path(temporary)
            for filename, data in files[:40]:
                path = root / safe_filename(filename)
                path.write_bytes(data)
                try:
                    body, metadata_author = load_catalog_source(path)
                except AutoPaperError:
                    body, metadata_author = "", ""
                excerpts.append(build_excerpt(Path(filename).name, body, metadata_author))
        with self._kb() as kb:
            authors = kb.list_authors()
            topics = kb.list_topics()
        llm = _open_llm(self.settings)
        try:
            result = suggest_many(excerpts, authors, topics, llm)
        finally:
            if llm is not None:
                llm.close()
        if len(files) > 40:
            result["message"] += " 一次最多阅读 40 篇的作者栏、摘要和结论，其余请手选。"
        return result

    def set_topic_parent(self, child_id: str, parent_id: str) -> dict:
        with self.lock, self._kb() as kb:
            kb.set_topic_parent(child_id, parent_id)
        return {"ok": True}

    def remove_topic_parent(self, child_id: str, parent_id: str) -> dict:
        with self.lock, self._kb() as kb:
            kb.remove_topic_parent(child_id, parent_id)
        return {"ok": True}

    def set_topic_related(self, left_id: str, right_id: str) -> dict:
        with self.lock, self._kb() as kb:
            kb.add_topic_related(left_id, right_id)
        return {"ok": True}

    def remove_topic_related(self, left_id: str, right_id: str) -> dict:
        with self.lock, self._kb() as kb:
            kb.remove_topic_related(left_id, right_id)
        return {"ok": True}

    def classify_topics(self) -> dict:
        llm = _open_llm(self.settings)
        try:
            with self.lock, self._kb() as kb:
                return reclassify_library(kb, llm)
        finally:
            if llm is not None:
                llm.close()

    def upload(self, fields: dict[str, list[str]], files: list[tuple[str, bytes]]) -> dict:
        if not files:
            raise AutoPaperError("请选择至少一篇 PDF、TXT 或 Markdown。")
        bibliography: list[dict] = []
        assignment_rows = _assignment_rows(fields)
        if assignment_rows is not None and len(assignment_rows) != len(files):
            raise AutoPaperError("作者和主题的篇数与文件数不一致")
        with tempfile.TemporaryDirectory(prefix="autopaper-upload-") as temporary:
            root = Path(temporary)
            used: set[str] = set()
            stored_names: list[str] = []
            for filename, data in files:
                name = safe_filename(filename)
                stem = Path(name).stem
                suffix = Path(name).suffix
                count = 2
                while name.casefold() in used:
                    name = safe_filename(f"{stem}-{count}{suffix}")
                    count += 1
                used.add(name.casefold())
                stored_names.append(name)
                (root / name).write_bytes(data)
            with self.lock, self._kb() as kb:
                planned = plan_uploads(root, kb, self.settings.data_dir)
                incoming = [item for item in planned if not item.skip]
                authors = []
                topics = []
                uploaded_papers = []
                by_name = None
                if incoming and assignment_rows is not None:
                    by_name = {}
                    for filename, stored_name, spec in zip((item[0] for item in files), stored_names, assignment_rows):
                        if Path(str(spec.get("name") or "")).name != Path(filename).name:
                            raise AutoPaperError(f"《{filename}》的建议和文件对不上")
                        try:
                            item_authors = resolve_authors(kb, _join_tokens(spec.get("authors")))
                            item_topics = resolve_topics(kb, _join_tokens(spec.get("topics")))
                        except AutoPaperError as exc:
                            raise AutoPaperError(f"{filename}：{exc}") from exc
                        by_name[stored_name] = (item_authors, item_topics)
                    authors = _unique_authors(author for pair in by_name.values() for author in pair[0])
                    topics = _unique_topics(topic for pair in by_name.values() for topic in pair[1])
                    planned = store_uploads(planned, kb, [], [], by_name)
                elif incoming:
                    author_raw = "；".join(fields.get("author") or [])
                    topic_raw = "；".join(fields.get("topic") or [])
                    authors = resolve_authors(kb, author_raw)
                    topics = resolve_topics(kb, topic_raw)
                    planned = store_uploads(planned, kb, authors, topics)
                if incoming:
                    uploaded_papers = [
                        kb.resolve_paper(item.paper_key) for item in planned if not item.skip and item.paper_key
                    ]
                    uploaded_papers = [paper for paper in uploaded_papers if paper is not None]
                    uploaded_papers, bibliography = recognize_uploaded(kb, uploaded_papers)
        result = {
            "uploaded": [
                {"title": item.title, "folder": str(item.folder) if item.folder else ""}
                for item in planned
                if not item.skip
            ],
            "skipped": [{"name": item.source.name, "reason": item.reason} for item in planned if item.skip],
            "authors": [author.name for author in authors],
            "topics": [topic.name for topic in topics],
            "notes": [],
            "note_status": "",
            "bibliography": [
                {"title": item["title"], "recognized": item["recognized"], "summary": format_recognized(item)}
                for item in bibliography
            ],
        }
        if not uploaded_papers:
            return result
        llm = _open_llm(self.settings)
        if llm is None:
            result["note_status"] = "未配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY，已归类但没有生成笔记。"
            return result
        try:
            with self._kb() as kb:
                papers = [kb.resolve_paper(paper.key) for paper in uploaded_papers]
                papers = [paper for paper in papers if paper is not None]
                written = note_papers(kb, self.settings.data_dir, llm, papers)
            result["notes"] = [str(path) for path in written]
            result["note_status"] = f"已生成 {len(written)} 份笔记。"
        except AutoPaperError as exc:
            result["note_status"] = f"论文已归类，笔记生成失败：{exc}"
        finally:
            llm.close()
        return result

    def _kb(self) -> KnowledgeBase:
        return KnowledgeBase(self.settings.database_path)


def create_server(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    app = LibraryApp(settings)
    handler = _handler_class(app)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = create_server(settings, host, port)
    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}/"
    print(f"AutoPaper 界面：{url}", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _handler_class(app: LibraryApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:
            try:
                self._get()
            except (AutoPaperError, NotFoundError) as exc:
                status = 404 if isinstance(exc, NotFoundError) else 400
                self._json({"error": str(exc)}, status=status)
            except Exception as exc:
                self._json({"error": str(exc)}, status=500)

        def do_DELETE(self) -> None:
            try:
                self._delete()
            except (AutoPaperError, NotFoundError) as exc:
                status = 404 if isinstance(exc, NotFoundError) else 400
                self._json({"error": str(exc)}, status=status)
            except Exception as exc:
                self._json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:
            try:
                self._post()
            except (AutoPaperError, NotFoundError) as exc:
                status = 404 if isinstance(exc, NotFoundError) else 400
                self._json({"error": str(exc)}, status=status)
            except Exception as exc:
                self._json({"error": str(exc)}, status=500)

        def _get(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query)
            if path in {"/", "/index.html"}:
                self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
                return
            if path.startswith("/assets/"):
                name = Path(path).name
                target = STATIC_DIR / name
                if target.parent != STATIC_DIR or not target.exists():
                    raise NotFoundError("没有这个页面资源")
                kind = "text/css; charset=utf-8" if target.suffix == ".css" else "text/javascript; charset=utf-8"
                self._file(target, kind)
                return
            if path == "/api/graph":
                self._json(app.graph())
                return
            if path == "/api/paper":
                self._json(app.paper(_one(query, "key")))
                return
            if path == "/api/file":
                target = app.file_path(_one(query, "key"))
                kind = "application/pdf" if target.suffix.lower() == ".pdf" else "text/plain; charset=utf-8"
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/api/synthesis":
                self._json(app.synthesis(_one(query, "author"), _one(query, "topic")))
                return
            if path == "/api/profile":
                self._json(app.read_profile(_one(query, "kind"), _one(query, "id")))
                return
            if path == "/api/highlights":
                self._json({"highlights": app.highlights(query.get("scope", [""])[0], query.get("id", [""])[0])})
                return
            if path == "/api/duplicates":
                self._json(app.duplicate_papers())
                return
            raise NotFoundError("没有这个地址")

        def _post(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length else b""
            if parsed.path == "/api/file/open":
                payload = _json_body(body)
                self._json(app.open_local(str(payload.get("key") or "")))
                return
            if parsed.path == "/api/annotate":
                payload = _json_body(body)
                self._json(app.annotate_original(str(payload.get("key") or "")))
                return
            if parsed.path == "/api/upload/suggest":
                _fields, files = parse_multipart(body, self.headers.get("Content-Type", ""))
                self._json(app.suggest_upload(files))
                return
            if parsed.path == "/api/upload":
                fields, files = parse_multipart(body, self.headers.get("Content-Type", ""))
                self._json(app.upload(fields, files))
                return
            if parsed.path == "/api/note":
                payload = _json_body(body)
                self._json(app.update_paper_note(str(payload.get("key") or "")))
                return
            if parsed.path == "/api/synthesis":
                payload = _json_body(body)
                self._json(app.update_synthesis(str(payload.get("author") or ""), str(payload.get("topic") or "")))
                return
            if parsed.path == "/api/highlights":
                self._json(app.save_highlights(_json_body(body)))
                return
            if parsed.path == "/api/ask":
                self._json(app.ask(_json_body(body)))
                return
            if parsed.path == "/api/profile":
                try:
                    payload = json.loads(body.decode("utf-8") or "{}")
                except json.JSONDecodeError as exc:
                    raise AutoPaperError("笔记请求不是 JSON") from exc
                self._json(app.write_profile(str(payload.get("kind") or ""), str(payload.get("id") or "")))
                return
            if parsed.path == "/api/biblio":
                payload = _json_body(body)
                self._json(app.fill_bibliography(str(payload.get("key") or ""), all_papers=bool(payload.get("all"))))
                return
            if parsed.path == "/api/paper-meta":
                payload = _json_body(body)
                self._json(
                    app.save_bibliography(
                        str(payload.get("key") or ""),
                        payload.get("year"),
                        str(payload.get("venue") or ""),
                        str(payload.get("doi") or ""),
                        str(payload.get("url") or ""),
                    )
                )
                return
            if parsed.path == "/api/topic-parent":
                payload = _json_body(body)
                self._json(app.set_topic_parent(str(payload.get("child") or ""), str(payload.get("parent") or "")))
                return
            if parsed.path == "/api/topic-related":
                payload = _json_body(body)
                self._json(app.set_topic_related(str(payload.get("left") or ""), str(payload.get("right") or "")))
                return
            if parsed.path == "/api/topics/classify":
                self._json(app.classify_topics())
                return
            if parsed.path == "/api/alias":
                payload = _json_body(body)
                author = str(payload.get("author") or "")
                add = [str(item) for item in payload.get("add") or []]
                remove = [str(item) for item in payload.get("remove") or []]
                self._json(app.update_alias(author, add, remove))
                return
            if parsed.path == "/api/duplicates":
                self._json(app.merge_duplicates())
                return
            raise NotFoundError("没有这个地址")

        def _delete(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if unquote(parsed.path) == "/api/highlights":
                self._json(app.remove_highlight(_one(query, "id")))
                return
            if unquote(parsed.path) == "/api/topic-parent":
                self._json(app.remove_topic_parent(_one(query, "child"), _one(query, "parent")))
                return
            if unquote(parsed.path) == "/api/topic-related":
                self._json(app.remove_topic_related(_one(query, "left"), _one(query, "right")))
                return
            path = unquote(parsed.path)
            if path == "/api/paper":
                self._json(app.remove_paper(_one(query, "key")))
                return
            if path == "/api/author":
                self._json(app.remove_author(_one(query, "id")))
                return
            if path == "/api/topic":
                self._json(app.remove_topic(_one(query, "id")))
                return
            if path == "/api/note":
                self._json(
                    app.remove_note(
                        query.get("scope", [""])[0],
                        query.get("id", [""])[0],
                        query.get("author", [""])[0],
                        query.get("topic", [""])[0],
                    )
                )
                return
            raise NotFoundError("没有这个地址")

        def _json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _file(self, path: Path, content_type: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _duplicate_message(groups: list) -> str:
    if not groups:
        return "没有发现重复论文。"
    count = sum(len(group) for group in groups)
    return f"发现 {len(groups)} 组、共 {count} 篇可能是同一篇。依据是题名或副标题、DOI，以及 arXiv、PubMed、OpenAlex、Semantic Scholar 标识。"


def launch_local_file(path: Path) -> None:
    """用系统默认程序打开论文库里的原件，阅读器保存时会写回这个文件。"""
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    command = ["open", os.fspath(path)] if sys.platform == "darwin" else ["xdg-open", os.fspath(path)]
    subprocess.Popen(command)


def _open_llm(settings: Settings) -> LLMClient | None:
    if settings.disable_llm or not settings.llm_api_key:
        return None
    return LLMClient(settings)


def _assignment_rows(fields: dict[str, list[str]]) -> list[dict] | None:
    raw_list = fields.get("assignments")
    if not raw_list or not str(raw_list[0]).strip():
        return None
    try:
        payload = json.loads(raw_list[0])
    except json.JSONDecodeError as exc:
        raise AutoPaperError("分篇作者和主题不是 JSON") from exc
    if not isinstance(payload, list):
        raise AutoPaperError("分篇作者和主题不是列表")
    rows: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            raise AutoPaperError("分篇作者和主题里有无法识别的一项")
        rows.append(item)
    return rows


def _join_tokens(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return ";".join(text for text in (str(item).strip() for item in raw) if text)
    return ""


def _unique_authors(items):
    chosen = []
    seen: set[str] = set()
    for author in items:
        if author.id in seen:
            continue
        seen.add(author.id)
        chosen.append(author)
    return chosen


def _unique_topics(items):
    chosen = []
    seen: set[str] = set()
    for topic in items:
        if topic.id in seen:
            continue
        seen.add(topic.id)
        chosen.append(topic)
    return chosen


def _json_body(body: bytes) -> dict:
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise AutoPaperError("请求不是 JSON") from exc
    if not isinstance(payload, dict):
        raise AutoPaperError("请求不是 JSON 对象")
    return payload


def _one(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name) or []
    text = values[0].strip() if values else ""
    if not text:
        raise AutoPaperError(f"缺少参数 {name}")
    return text
