from pathlib import Path

from autopaper.config import Settings
from autopaper.models import Paper, ScoredPaper, Topic
from autopaper.storage.db import KnowledgeBase
from autopaper.storage.export import export_topic, render_bibtex


def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path)


def test_database_merges_records_onto_the_canonical_key(tmp_path: Path):
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        first = kb.upsert_paper(
            Paper(
                title="Graph Neural Networks for Time Series Forecasting",
                year=2023,
                arxiv_id="2401.00001",
                source="arxiv",
            )
        )
        assert first == "arxiv:2401.00001"
        topic = Topic(id="gnn", name="GNN", query="gnn", sources=["arxiv"], min_score=50)
        kb.save_topic(topic)
        paper = kb.resolve_paper(first)
        kb.link_topic_paper(
            "gnn",
            ScoredPaper(paper, 80, None, 80, "相关", "candidate"),
            paper.key,
        )
        author = kb.create_author("Ada")
        kb.link_author_paper(author.id, first)
        kb.save_reading(first, {"title": "t"}, "# t\n")
        kb.set_local_path(first, "D:/papers/a.pdf")

        second = kb.upsert_paper(
            Paper(
                title="Graph Neural Networks for Time Series Forecasting",
                doi="10.1000/xyz",
                abstract="hello abstract",
                citation_count=5,
                source="crossref",
            )
        )
        assert second == "doi:10.1000/xyz"
        stored = kb.resolve_paper(second)
        assert stored.doi == "10.1000/xyz"
        assert stored.arxiv_id == "2401.00001"
        assert stored.abstract == "hello abstract"
        assert stored.local_path == "D:/papers/a.pdf"
        assert kb.list_topic_papers("gnn")[0].paper.key == second
        assert kb.get_reading(second)["markdown"] == "# t\n"
        assert kb.authors_of_paper(second)[0].name == "Ada"

        reversed_doi = kb.upsert_paper(
            Paper(title="Stable Canonical Keys for Exports", doi="10.1000/rev", source="crossref")
        )
        reversed_arxiv = kb.upsert_paper(
            Paper(title="Stable Canonical Keys for Exports", arxiv_id="2402.00002", source="arxiv")
        )
        assert reversed_doi == reversed_arxiv == "doi:10.1000/rev"

        left = kb.upsert_paper(Paper(title="Introduction", doi="10.1/a", source="crossref"))
        right = kb.upsert_paper(Paper(title="Introduction", doi="10.1/b", source="crossref"))
        assert left != right
        kb.link_topic_paper(
            "gnn",
            ScoredPaper(kb.resolve_paper(left), 10, None, 10, "偏低", "held"),
            left,
        )
        assert len(kb.list_topic_papers("gnn", include_held=False)) == 1
        assert kb.search_papers("hello abstract")[0].key == second
        kb.delete_topic("gnn")
        assert kb.list_topic_papers("gnn") == []
        assert kb.resolve_paper(second) is not None


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
