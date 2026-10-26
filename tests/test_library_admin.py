from pathlib import Path

from autopaper.library import plan_uploads, remove_paper_files, remove_paper_note
from autopaper.models import Paper, ScoredPaper, Topic
from autopaper.pipeline.dedup import deduplicate, same_paper
from autopaper.storage.db import KnowledgeBase


SHORT = "Psychometrics in Behavioral Software Engineering"
LONG = "Psychometrics in Behavioral Software Engineering: A Methodological Introduction with Guidelines"


def test_subtitle_and_doi_count_as_the_same_paper_but_conflicting_dois_do_not():
    short = Paper(title=SHORT)
    long = Paper(title=LONG, doi="10.5555/12345678")
    other = Paper(title=SHORT, doi="10.5555/87654321")
    assert same_paper(short, long, 92)
    assert not same_paper(long, other, 92)
    merged = deduplicate([short, long], 92)
    assert len(merged) == 1
    assert merged[0].title == LONG
    assert merged[0].doi == "10.5555/12345678"
    kept = deduplicate([long, other], 92)
    assert len(kept) == 2


def test_library_merge_keeps_the_longer_title_links_and_note(tmp_path: Path):
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        short_key = kb.upsert_paper(Paper(title=SHORT, source="local"))
        long_key = kb.upsert_paper(Paper(title=LONG, doi="10.5555/12345678", abstract="longer", source="local"))
        author = kb.create_author("Ada Lovelace")
        topic = kb.save_topic(Topic(id="scale", name="Scale", query="scale", sources=["local"]))
        kb.link_author_paper(author.id, short_key)
        kb.link_topic_paper(topic.id, ScoredPaper(Paper(title=SHORT), 80, None, 80, "短题名", "kept"), short_key)
        kb.save_reading(long_key, {"one_sentence": "全称"}, "全称笔记")
        kb.set_local_path(short_key, str(tmp_path / "short.pdf"))
        groups = kb.duplicate_groups()
        assert len(groups) == 1
        merged = kb.merge_same_papers()
        assert len(merged) == 1
        assert merged[0]["title"] == LONG
        papers = kb.list_papers()
        assert len(papers) == 1
        kept = papers[0]
        assert kept.doi == "10.5555/12345678"
        assert kept.local_path.endswith("short.pdf")
        assert [item.id for item in kb.authors_of_paper(kept.key)] == [author.id]
        assert kb.get_reading(kept.key)["markdown"] == "全称笔记"
        assert any(item.paper.key == kept.key for item in kb.list_topic_papers(topic.id))


def test_upload_skips_subtitle_and_doi_matches(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / f"{SHORT}.txt").write_text("a short filename for the same paper", encoding="utf-8")
    (inbox / "renamed-paper.txt").write_text("The DOI is 10.5555/12345678 in the text.", encoding="utf-8")
    (inbox / "different-paper.txt").write_text("DOI 10.5555/99999999 belongs to another paper.", encoding="utf-8")
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        kb.upsert_paper(Paper(title=LONG, doi="10.5555/12345678"))
        planned = plan_uploads(inbox, kb, tmp_path)
        reasons = {item.source.name: item.reason for item in planned if item.skip}
        assert SHORT + ".txt" in reasons
        assert "同一篇" in reasons[SHORT + ".txt"]
        assert "renamed-paper.txt" in reasons
        assert "different-paper.txt" not in reasons


def test_delete_paper_author_topic_and_note(tmp_path: Path):
    library = tmp_path / "library" / "alpha-paper"
    library.mkdir(parents=True)
    pdf = library / "Alpha Paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    (library / "note.md").write_text("笔记", encoding="utf-8")
    (library / "title.txt").write_text("Alpha Paper\n", encoding="utf-8")
    other = tmp_path / "library" / "beta-paper"
    other.mkdir()
    (other / "Beta Paper.pdf").write_bytes(b"%PDF-1.4")
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        alpha = kb.upsert_paper(Paper(title="Alpha Paper", source="local"))
        beta = kb.upsert_paper(Paper(title="Beta Paper", source="local"))
        kb.set_local_path(alpha, str(pdf))
        kb.set_local_path(beta, str(other / "Beta Paper.pdf"))
        author = kb.create_author("Ada Lovelace")
        topic = kb.save_topic(Topic(id="scale", name="Scale", query="scale", sources=["local"]))
        kb.link_author_paper(author.id, alpha)
        kb.save_reading(alpha, {}, "笔记")
        paper = kb.resolve_paper(alpha)
        assert paper is not None
        kb.delete_reading(alpha)
        remove_paper_note(paper)
        assert kb.get_reading(alpha) is None
        assert not (library / "note.md").exists()
        assert pdf.exists()
        remove_paper_files(tmp_path, paper, kb)
        kb.delete_paper(alpha)
        assert kb.resolve_paper(alpha) is None
        assert not library.exists()
        assert other.exists()
        kb.delete_author(author.id)
        assert kb.find_author("Ada Lovelace") is None
        assert kb.resolve_paper(beta) is not None
        kb.delete_topic_record(topic.id)
        assert kb.resolve_paper(beta) is not None
