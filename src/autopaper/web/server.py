from __future__ import annotations

import json
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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
from autopaper.catalog import require_existing_author, resolve_authors, resolve_topics
from autopaper.config import Settings
from autopaper.errors import AutoPaperError, NotFoundError
from autopaper.library import plan_uploads, safe_filename, store_uploads
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

    def remove_highlight(self, ident: str) -> dict:
        with self.lock, self._kb() as kb:
            kb.delete_highlight(ident)
            rows = kb.list_highlights()
        write_highlight_library(self.settings.data_dir, rows)
        return {"ok": True}

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

    def upload(self, fields: dict[str, list[str]], files: list[tuple[str, bytes]]) -> dict:
        if not files:
            raise AutoPaperError("请选择至少一篇 PDF、TXT 或 Markdown。")
        with tempfile.TemporaryDirectory(prefix="autopaper-upload-") as temporary:
            root = Path(temporary)
            used: set[str] = set()
            for filename, data in files:
                name = safe_filename(filename)
                stem = Path(name).stem
                suffix = Path(name).suffix
                count = 2
                while name.casefold() in used:
                    name = safe_filename(f"{stem}-{count}{suffix}")
                    count += 1
                used.add(name.casefold())
                (root / name).write_bytes(data)
            with self.lock, self._kb() as kb:
                planned = plan_uploads(root, kb, self.settings.data_dir)
                incoming = [item for item in planned if not item.skip]
                authors = []
                topics = []
                uploaded_papers = []
                if incoming:
                    author_raw = "；".join(fields.get("author") or [])
                    topic_raw = "；".join(fields.get("topic") or [])
                    authors = resolve_authors(kb, author_raw)
                    topics = resolve_topics(kb, topic_raw)
                    planned = store_uploads(planned, kb, authors, topics)
                    uploaded_papers = [
                        kb.resolve_paper(item.paper_key) for item in planned if not item.skip and item.paper_key
                    ]
                    uploaded_papers = [paper for paper in uploaded_papers if paper is not None]
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
            raise NotFoundError("没有这个地址")

        def _post(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length else b""
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
            if parsed.path == "/api/profile":
                try:
                    payload = json.loads(body.decode("utf-8") or "{}")
                except json.JSONDecodeError as exc:
                    raise AutoPaperError("笔记请求不是 JSON") from exc
                self._json(app.write_profile(str(payload.get("kind") or ""), str(payload.get("id") or "")))
                return
            if parsed.path == "/api/alias":
                payload = _json_body(body)
                author = str(payload.get("author") or "")
                add = [str(item) for item in payload.get("add") or []]
                remove = [str(item) for item in payload.get("remove") or []]
                self._json(app.update_alias(author, add, remove))
                return
            raise NotFoundError("没有这个地址")

        def _delete(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if unquote(parsed.path) == "/api/highlights":
                self._json(app.remove_highlight(_one(query, "id")))
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


def _open_llm(settings: Settings) -> LLMClient | None:
    if settings.disable_llm or not settings.llm_api_key:
        return None
    return LLMClient(settings)


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
