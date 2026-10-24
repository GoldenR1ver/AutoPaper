from pathlib import Path

from autopaper.analysis.biblio import extract_from_parts, fill_paper, replace_meta_block
from autopaper.catalog import resolve_authors, resolve_topics
from autopaper.library import plan_uploads, store_uploads
from autopaper.models import Paper
from autopaper.storage.db import KnowledgeBase


def test_doi_year_beats_study_years_and_placeholders():
    text = """
    The study ran from 2017 to 2019.
    CHASE 2019
    © 2019
    https://doi.org/10.1145/1122445.1122456
    10.1109/MS.2019
    10.1073/pnas.
    https://doi.org/10.1109/CHASE52884.2021.00018
    """
    found = extract_from_parts("Approaches to Diversifying the Programmer", "", text)
    assert found.doi == "10.1109/chase52884.2021.00018"
    assert found.year == 2021
    assert found.venue == "CHASE"
    assert found.url == "https://doi.org/10.1109/chase52884.2021.00018"


def test_proceedings_line_beats_filename_and_template_doi():
    text = (
        "Hana Frluckaj. 2022. Gender and Participation. "
        "In CSCW '22: ACM Conference, Nov 8-22, 2022. "
        "https://doi.org/10.1145/1122445.1122456"
    )
    found = extract_from_parts("frluckaj-gender-cscw2023", "", text)
    assert found.year == 2022
    assert found.venue == "CSCW"
    assert found.doi == ""
    assert found.url == ""


def test_placeholder_doi_falls_back_to_arxiv_and_venue_year():
    text = """
    ICSE '26, Rio de Janeiro, Brazil
    ©2025 Copyright held by the owner/author(s).
    https://doi.org/10.1145/nnnnnnn.nnnnnnn
    arXiv:2512.00855v1 [cs.SE] 30 Nov 2025
    """
    found = extract_from_parts("The Software Infrastructure Attitude Scale (SIAS)", "", text)
    assert found.doi == ""
    assert found.year == 2026
    assert found.venue == "ICSE"
    assert found.url == "https://arxiv.org/abs/2512.00855"


def test_filename_year_is_not_treated_as_chase_venue():
    found = extract_from_parts("chase2024organization", "", "A story that also mentions 2021.")
    assert found.year == 2024
    assert found.venue == ""
    assert found.doi == ""


def test_filename_supplies_year_and_venue_when_the_page_has_neither():
    found = extract_from_parts("zhao_2023_seis", "", "Gender representation among contributors.")
    assert found.year == 2023
    assert found.venue == "ICSE-SEIS"
    assert found.doi == ""


def test_fill_keeps_manual_values_and_refreshes_the_note_header(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "Seeking Measures.txt").write_text(
        "CHASE ’22, May 21–29, 2022, Pittsburgh, PA, USA © 2022 Copyright\n"
        "https://doi.org/10.1145/3528579.3529169\n",
        encoding="utf-8",
    )
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        planned = plan_uploads(inbox, kb, tmp_path)
        store_uploads(planned, kb, resolve_authors(kb, "Huilian Qiu"), resolve_topics(kb, "Science of Science"))
        paper = kb.list_local_papers()[0]
        original = "\n".join(
            [
                "# Seeking Measures",
                "",
                "模型：测试",
                "",
                "标题：Seeking Measures",
                "作者：Huilian Qiu",
                "年份：未知",
                "期刊或会议：未知",
                "DOI：无",
                "链接：无",
                "",
                "## 一句话",
                "",
                "保留这段。",
                "",
            ]
        )
        kb.save_reading(paper.key, {"title": paper.title}, original)
        created = kb.get_reading(paper.key)["created_at"]
        filled = fill_paper(kb, paper, only_missing=True)
        assert filled["year"] == 2022
        assert filled["venue"] == "CHASE"
        assert filled["doi"] == "10.1145/3528579.3529169"
        reading = kb.get_reading(paper.key)
        assert reading["created_at"] == created
        assert "年份：2022" in reading["markdown"]
        assert "期刊或会议：CHASE" in reading["markdown"]
        assert "保留这段。" in reading["markdown"]
        note = Path(paper.local_path).parent / "note.md"
        note.write_text(reading["markdown"], encoding="utf-8")
        stored = kb.resolve_paper(paper.key)
        stored.year = 1999
        kb.update_bibliography(stored)
        again = fill_paper(kb, kb.resolve_paper(paper.key), only_missing=True)
        assert again["year"] == 1999
        assert "year" not in again["filled"]


def test_replace_meta_block_leaves_the_body():
    paper = Paper(title="Seeking Measures", authors=["Huilian Qiu"], year=2022, venue="CHASE")
    updated = replace_meta_block("标题：旧\n作者：旧\n年份：未知\n期刊或会议：未知\nDOI：无\n链接：无\n\n正文\n", paper)
    assert updated.startswith("标题：Seeking Measures\n作者：Huilian Qiu\n年份：2022\n")
    assert updated.endswith("\n\n正文\n")
