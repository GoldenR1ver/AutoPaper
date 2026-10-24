from pathlib import Path

from autopaper.config import Settings
from autopaper.models import Paper, ScoredPaper, Topic
from autopaper.storage.db import KnowledgeBase
from autopaper.storage.export import export_topic, render_bibtex


def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path)


def test_database_merges_records_and_keeps_the_first_key(tmp_path: Path):
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        first = kb.upsert_paper(
            Paper(
                title="Graph Neural Networks for Time Series Forecasting",
                year=2023,
                arxiv_id="2401.00001",
                source="arxiv",
            )
        )
        second = kb.upsert_paper(
            Paper(
                title="Graph Neural Networks for Time Series Forecasting",
                doi="10.1000/xyz",
                abstract="hello abstract",
                citation_count=5,
                source="crossref",
            )
        )
        assert first == second
        stored = kb.resolve_paper(first)
        assert stored.doi == "10.1000/xyz"
        assert stored.arxiv_id == "2401.00001"
        assert stored.abstract == "hello abstract"

        left = kb.upsert_paper(Paper(title="Introduction", doi="10.1/a", source="crossref"))
        right = kb.upsert_paper(Paper(title="Introduction", doi="10.1/b", source="crossref"))
        assert left != right

        topic = Topic(id="gnn", name="GNN", query="gnn", sources=["arxiv"], min_score=50)
        kb.save_topic(topic)
        paper = kb.resolve_paper(first)
        kb.link_topic_paper(
            "gnn",
            ScoredPaper(paper, 80, None, 80, "相关", "candidate"),
            paper.key,
        )
        kb.link_topic_paper(
            "gnn",
            ScoredPaper(paper, 10, None, 10, "偏低", "held"),
            kb.resolve_paper(left).key,
        )
        assert len(kb.list_topic_papers("gnn", include_held=False)) == 1
        assert kb.search_papers("hello abstract")[0].key == first
        kb.delete_topic("gnn")
        assert kb.list_topic_papers("gnn") == []
        assert kb.resolve_paper(first) is not None


def test_exports_escape_bibtex_and_write_utf8_csv(tmp_path: Path):
    paper = Paper(
        title="Graphs & Time_Series",
        authors=["Ada Lovelace"],
        year=2024,
        doi="10.1000/abc",
        venue="JMLR",
        key="doi:10.1000/abc",
    )
    bib = render_bibtex([paper])
    assert r"Graphs \& Time\_Series" in bib
    assert "author = {Ada Lovelace}" in bib
    item = ScoredPaper(paper, 70, None, 70, "因为相关", "candidate")
    written = export_topic(tmp_path, "GNN", [item], {"csv", "json", "bib", "md"})
    assert written["csv"].read_bytes().startswith(b"\xef\xbb\xbf")
    assert "引用列表" in written["md"].read_text(encoding="utf-8")
    assert "10.1000/abc" in written["json"].read_text(encoding="utf-8")
