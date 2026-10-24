from pathlib import Path

from autopaper.analysis.notes import write_highlight_library
from autopaper.catalog import resolve_authors, resolve_topics
from autopaper.library import plan_uploads, store_uploads
from autopaper.storage.db import KnowledgeBase


def test_highlights_remain_when_the_note_is_rewritten(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "Alpha Paper.txt").write_text("promotional language", encoding="utf-8")
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        planned = plan_uploads(inbox, kb, tmp_path)
        store_uploads(planned, kb, resolve_authors(kb, "Huilian Qiu"), resolve_topics(kb, "Science of Science"))
        paper = kb.list_local_papers()[0]
        kb.save_reading(paper.key, {"title": "Alpha Paper"}, "# Alpha\n\n第一段结论。\n")
        saved = kb.add_highlight(
            scope="paper",
            scope_id=paper.key,
            scope_label="Alpha Paper",
            kind="highlight",
            color="blue",
            paragraph="第一段结论。",
            selected_text="结论",
            comment="",
        )
        note = kb.add_highlight(
            scope="author",
            scope_id=kb.authors_of_paper(paper.key)[0].id,
            scope_label="Huilian Qiu",
            kind="annotation",
            color="",
            paragraph="第一段结论。",
            selected_text="第一段结论。",
            comment="这里要回头核对样本。",
        )
        kb.save_reading(paper.key, {"title": "Alpha Paper"}, "# Alpha\n\n改写后的结论。\n")
        rows = kb.list_highlights()
        assert [item["id"] for item in kb.list_highlights("paper", paper.key)] == [saved["id"]]
        assert kb.list_highlights("paper", paper.key)[0]["paragraph"] == "第一段结论。"
        assert "改写后的结论" in kb.get_reading(paper.key)["markdown"]
        assert note["comment"] == "这里要回头核对样本。"
    library = write_highlight_library(tmp_path, rows)
    text = library.read_text(encoding="utf-8")
    assert "我的笔记库" in text
    assert "第一段结论" in text
    assert "这里要回头核对样本" in text
