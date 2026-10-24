import json
import threading
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from autopaper.config import load_settings
from autopaper.web.graph import build_library_graph
from autopaper.web.server import create_server, parse_multipart


def _multipart(fields: dict[str, str], files: list[tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = "AutoPaperBoundary"
    chunks = []
    for name, value in fields.items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8")
        )
    for filename, data in files:
        header = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            "Content-Type: text/plain\r\n\r\n"
        ).encode("utf-8")
        chunks.append(header + data + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def test_parse_multipart_keeps_file_bytes():
    body, content_type = _multipart({"author": "Huilian Qiu"}, [("Alpha Paper.txt", b"alpha\r\n")])
    fields, files = parse_multipart(body, content_type)
    assert fields["author"] == ["Huilian Qiu"]
    assert files == [("Alpha Paper.txt", b"alpha\r\n")]


def test_graph_links_similar_notes(tmp_path: Path):
    from autopaper.catalog import resolve_authors, resolve_topics
    from autopaper.library import plan_uploads, store_uploads
    from autopaper.storage.db import KnowledgeBase

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "Alpha Paper.txt").write_text("promotional language and grant success", encoding="utf-8")
    (inbox / "Beta Study.txt").write_text("promotional language and scientific impact", encoding="utf-8")
    (inbox / "Unrelated Pasta.txt").write_text("boiling water and wheat shapes", encoding="utf-8")
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        planned = plan_uploads(inbox, kb, tmp_path)
        authors = resolve_authors(kb, "Huilian Qiu")
        topics = resolve_topics(kb, "Science of Science")
        store_uploads(planned, kb, authors, topics)
        saved = {paper.title: paper for paper in kb.list_local_papers()}
        notes = {
            "Alpha Paper": "推广性语言与资助成功有关",
            "Beta Study": "推广性语言与科学影响力有关",
            "Unrelated Pasta": "煮面条时的水温",
        }
        for title, sentence in notes.items():
            kb.save_reading(
                saved[title].key,
                {"title": title, "one_sentence": sentence, "field": sentence, "problem": sentence, "method": sentence},
                f"# {title}\n\n{sentence}\n",
            )
        graph = build_library_graph(kb, tmp_path)
        keys = {item["title"]: item["key"] for item in graph["papers"]}

        def weight(left: str, right: str) -> int:
            pair = {keys[left], keys[right]}
            return next(edge["weight"] for edge in graph["edges"] if {edge["source"], edge["target"]} == pair)

        assert weight("Alpha Paper", "Beta Study") > weight("Alpha Paper", "Unrelated Pasta")
    assert graph["authors"][0]["label"] == "Huilian Qiu"
    assert any(item["has_synthesis"] is False for item in graph["syntheses"])


@pytest.fixture()
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOPAPER_CONFIG", raising=False)
    monkeypatch.setenv("AUTOPAPER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    settings = load_settings()
    httpd = create_server(settings, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base
    finally:
        httpd.shutdown()
        httpd.server_close()


def _json(url: str, data: bytes | None = None, content_type: str | None = None, method: str | None = None):
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_interface_uploads_reads_and_edits_aliases(server: str):
    page = urlopen(server + "/").read().decode("utf-8")
    assert "按相关度" in page
    assert "我的笔记库" in page
    assert "更新笔记" in page
    body, content_type = _multipart(
        {"author": "Huilian Qiu", "topic": "Science of Science"},
        [("Alpha Paper.txt", "推广性语言与资助。".encode("utf-8"))],
    )
    status_code, payload = _json(server + "/api/upload", body, content_type)
    assert status_code == 200
    assert payload["uploaded"][0]["title"] == "Alpha Paper"
    _, again = _json(server + "/api/upload", body, content_type)
    assert again["skipped"]
    status_code, graph = _json(server + "/api/graph")
    assert graph["papers"][0]["title"] == "Alpha Paper"
    key = graph["papers"][0]["key"]
    status_code, paper = _json(server + "/api/paper?key=" + quote(key, safe=""))
    assert "还没有笔记" in paper["markdown"] or paper["paper"]["title"] == "Alpha Paper"
    assert "updated_at" in paper
    assert "note_updated_at" in graph["papers"][0]
    status_code, author = _json(
        server + "/api/alias",
        json.dumps({"author": "Huilian Qiu", "add": ["邱惠莲"], "remove": []}).encode("utf-8"),
        "application/json",
    )
    assert author["label"] == "Huilian Qiu（邱惠莲）"
    status_code, saved = _json(
        server + "/api/highlights",
        json.dumps({
            "scope": "paper",
            "scope_id": key,
            "scope_label": "Alpha Paper",
            "kind": "highlight",
            "color": "yellow",
            "items": [{"paragraph": "推广性语言与资助成功有关。", "selected_text": "推广性语言"}],
        }).encode("utf-8"),
        "application/json",
    )
    assert status_code == 200
    highlight_id = saved["highlights"][0]["id"]
    _, listed = _json(server + "/api/highlights?scope=paper&id=" + quote(key, safe=""))
    assert listed["highlights"][0]["paragraph"] == "推广性语言与资助成功有关。"
    _, library = _json(server + "/api/highlights")
    assert any(item["id"] == highlight_id for item in library["highlights"])
    status_code, removed = _json(server + "/api/highlights?id=" + highlight_id, method="DELETE")
    assert removed["ok"] is True
    _, empty = _json(server + "/api/highlights")
    assert empty["highlights"] == []
