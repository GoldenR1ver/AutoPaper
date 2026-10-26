import json
import threading
from pathlib import Path
from urllib.parse import quote
from urllib.error import HTTPError
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
    _, asked = _json(
        server + "/api/highlights",
        json.dumps({
            "scope": "paper",
            "scope_id": key,
            "scope_label": "Alpha Paper",
            "kind": "annotation",
            "color": "blue",
            "comment": "追问：这段在说什么\n这段在讲推广性语言。",
            "items": [{"paragraph": "推广性语言与资助成功有关。", "selected_text": "推广性语言"}],
        }).encode("utf-8"),
        "application/json",
    )
    assert asked["highlights"][0]["color"] == "blue"
    assert asked["highlights"][0]["comment"].startswith("追问：")
    try:
        _json(
            server + "/api/ask",
            json.dumps({
                "scope": "paper",
                "scope_id": key,
                "question": "meaning",
                "items": [{"paragraph": "推广性语言与资助成功有关。", "selected_text": "推广性语言"}],
                "note": "推广性语言与资助成功有关。",
            }).encode("utf-8"),
            "application/json",
        )
    except HTTPError as exc:
        assert exc.code == 400
        assert "API_KEY" in json.loads(exc.read().decode("utf-8"))["error"]
    else:
        raise AssertionError("没有密钥时不应完成追问")


def test_bibliography_is_recognized_on_upload_and_can_be_edited(server: str):
    page = urlopen(server + "/").read().decode("utf-8")
    assert "补全题录" in page
    script = urlopen(server + "/assets/app.js").read().decode("utf-8")
    assert "保存题录" in script
    assert "从原文补全" in script
    body, content_type = _multipart(
        {"author": "Huilian Qiu", "topic": "Science of Science"},
        [(
            "Seeking Measures.txt",
            "CHASE '22, May 21–29, 2022 © 2022 Copyright\nhttps://doi.org/10.1145/3528579.3529169\n".encode("utf-8"),
        )],
    )
    status_code, uploaded = _json(server + "/api/upload", body, content_type)
    assert status_code == 200
    recognized = uploaded["bibliography"][0]["recognized"]
    assert recognized["year"] == 2022
    assert recognized["venue"] == "CHASE"
    assert recognized["doi"] == "10.1145/3528579.3529169"
    _, graph = _json(server + "/api/graph")
    key = graph["papers"][0]["key"]
    assert graph["papers"][0]["year"] == 2022
    status_code, saved = _json(
        server + "/api/paper-meta",
        json.dumps({"key": key, "year": "1999", "venue": "手改会议", "doi": recognized["doi"], "url": ""}).encode("utf-8"),
        "application/json",
    )
    assert saved["year"] == 1999
    assert saved["venue"] == "手改会议"
    assert saved["url"] == "https://doi.org/10.1145/3528579.3529169"
    _, filled = _json(
        server + "/api/biblio",
        json.dumps({"key": key}).encode("utf-8"),
        "application/json",
    )
    assert filled["papers"][0]["year"] == 1999
    assert "year" not in filled["papers"][0]["filled"]
    _, paper = _json(server + "/api/paper?key=" + quote(key, safe=""))
    assert paper["paper"]["venue"] == "手改会议"


def test_upload_suggests_known_authors_and_topics_and_classify_adds_a_match(server: str):
    page = urlopen(server + "/").read().decode("utf-8")
    assert "再分类" in page
    assert "每篇会单独根据作者栏、摘要和结论" in page
    body, content_type = _multipart(
        {"author": "Huilian Qiu", "topic": "Science of Science"},
        [("Alpha Paper.txt", "promotional language in grant applications".encode("utf-8"))],
    )
    _json(server + "/api/upload", body, content_type)
    other_body, other_type = _multipart(
        {"author": "Huilian Qiu", "topic": "Promotional Language"},
        [("Beta Paper.txt", "beta".encode("utf-8"))],
    )
    _json(server + "/api/upload", other_body, other_type)
    suggest_body, suggest_type = _multipart(
        {},
        [
            ("Gamma.txt", "Huilian Qiu studies promotional language".encode("utf-8")),
            ("Delta.txt", "Science of Science with semantics".encode("utf-8")),
        ],
    )
    status_code, suggested = _json(server + "/api/upload/suggest", suggest_body, suggest_type)
    assert status_code == 200
    assert suggested["papers"][0]["name"] == "Gamma.txt"
    assert any(item["name"] == "Huilian Qiu" for item in suggested["papers"][0]["authors"])
    assert any(item["name"] == "Promotional Language" for item in suggested["papers"][0]["topics"])
    assert suggested["papers"][1]["authors"] == []
    assert any(item["name"] == "Science of Science" for item in suggested["papers"][1]["topics"])
    assert not any(item["name"] == "Huilian Qiu" for item in suggested["papers"][1]["authors"])
    assignments = json.dumps(
        [
            {"name": "One.txt", "authors": ["Huilian Qiu"], "topics": ["Science of Science"]},
            {"name": "Two.txt", "authors": ["Ada Lovelace"], "topics": ["Promotional Language"]},
        ]
    )
    assigned_body, assigned_type = _multipart(
        {"assignments": assignments},
        [("One.txt", "one".encode("utf-8")), ("Two.txt", "two".encode("utf-8"))],
    )
    status_code, assigned = _json(server + "/api/upload", assigned_body, assigned_type)
    assert status_code == 200
    _, graph = _json(server + "/api/graph")
    one = next(item for item in graph["papers"] if item["title"] == "One")
    two = next(item for item in graph["papers"] if item["title"] == "Two")
    assert [item["label"] for item in one["authors"]] == ["Huilian Qiu"]
    assert [item["name"] for item in one["topics"] if not item.get("inherited")] == ["Science of Science"]
    assert [item["label"] for item in two["authors"]] == ["Ada Lovelace"]
    assert [item["name"] for item in two["topics"] if not item.get("inherited")] == ["Promotional Language"]
    _, graph = _json(server + "/api/graph")
    science = next(item for item in graph["topics"] if item["name"] == "Science of Science")
    promo = next(item for item in graph["topics"] if item["name"] == "Promotional Language")
    status_code, linked = _json(
        server + "/api/topic-parent",
        json.dumps({"child": promo["id"], "parent": science["id"]}).encode("utf-8"),
        "application/json",
    )
    assert status_code == 200
    _, graph = _json(server + "/api/graph")
    promo = next(item for item in graph["topics"] if item["name"] == "Promotional Language")
    assert promo["parents"][0]["name"] == "Science of Science"
    science = next(item for item in graph["topics"] if item["name"] == "Science of Science")
    assert any(item["title"].startswith("Beta") for item in graph["papers"] if any(topic["id"] == science["id"] and topic["inherited"] for topic in item["topics"]))
    status_code, classified = _json(server + "/api/topics/classify", b"{}", "application/json")
    assert status_code == 200
    assert classified["added"] >= 1


def test_open_file_menu_can_launch_the_local_reader(server: str, monkeypatch: pytest.MonkeyPatch):
    page = urlopen(server + "/").read().decode("utf-8")
    assert "从网页打开原文" in page
    assert "用本机阅读器打开" in page
    opened: list[Path] = []
    monkeypatch.setattr("autopaper.web.server.launch_local_file", lambda path: opened.append(Path(path)))
    body, content_type = _multipart(
        {"author": "Huilian Qiu", "topic": "Science of Science"},
        [("Alpha Paper.txt", "原文正文".encode("utf-8"))],
    )
    _json(server + "/api/upload", body, content_type)
    _, graph = _json(server + "/api/graph")
    key = graph["papers"][0]["key"]
    status_code, payload = _json(
        server + "/api/file/open",
        json.dumps({"key": key}).encode("utf-8"),
        "application/json",
    )
    assert status_code == 200 and payload["ok"] is True
    assert opened[0].is_file()
    assert "library" in opened[0].parts
    file_bytes = urlopen(server + "/api/file?key=" + quote(key, safe="")).read()
    assert file_bytes == "原文正文".encode("utf-8")


def test_annotate_endpoint_rejects_non_pdf_and_can_write_a_pdf(server: str, monkeypatch: pytest.MonkeyPatch):
    page = urlopen(server + "/").read().decode("utf-8")
    assert "自动批注原文" in page
    body, content_type = _multipart(
        {"author": "Huilian Qiu", "topic": "Science of Science"},
        [("Alpha Paper.txt", "原文正文".encode("utf-8"))],
    )
    _json(server + "/api/upload", body, content_type)
    _, graph = _json(server + "/api/graph")
    text_key = graph["papers"][0]["key"]
    assert graph["papers"][0]["file_suffix"] == ".txt"
    try:
        _json(server + "/api/annotate", json.dumps({"key": text_key}).encode("utf-8"), "application/json")
    except HTTPError as exc:
        assert exc.code == 400
        assert "PDF" in json.loads(exc.read().decode("utf-8"))["error"]
    else:
        raise AssertionError("文本文件不应写入 PDF 批注")

    from io import BytesIO

    from pypdf import PdfWriter

    pdf_buffer = BytesIO()
    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=612, height=792)
    pdf_writer.write(pdf_buffer)
    pdf_body, pdf_type = _multipart(
        {"author": "Huilian Qiu", "topic": "Science of Science"},
        [("Beta Paper.pdf", pdf_buffer.getvalue())],
    )
    _json(server + "/api/upload", pdf_body, pdf_type)
    _, graph = _json(server + "/api/graph")
    pdf_paper = next(item for item in graph["papers"] if item["title"].startswith("Beta"))
    assert pdf_paper["file_suffix"] == ".pdf"

    class Dummy:
        def close(self) -> None:
            return None

    seen: dict[str, Path] = {}

    def fake_annotate(path, llm):
        from autopaper.analysis.pdfnotes import AnnotateResult

        seen["path"] = Path(path)
        return AnnotateResult(path=Path(path), added=2, by_kind={"keyword": 2})

    monkeypatch.setattr("autopaper.web.server._open_llm", lambda settings: Dummy())
    monkeypatch.setattr("autopaper.web.server.annotate_pdf", fake_annotate)
    status_code, payload = _json(
        server + "/api/annotate",
        json.dumps({"key": pdf_paper["key"]}).encode("utf-8"),
        "application/json",
    )
    assert status_code == 200
    assert payload["added"] == 2
    assert seen["path"].suffix.lower() == ".pdf"
    assert "library" in seen["path"].parts
