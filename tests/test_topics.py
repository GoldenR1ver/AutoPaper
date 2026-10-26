from pathlib import Path

import pytest

from autopaper.analysis.classify import reclassify_library, suggest_catalog, suggest_many
from autopaper.analysis.excerpt import build_excerpt
from autopaper.errors import AutoPaperError
from autopaper.models import Author, Paper, ScoredPaper, Topic
from autopaper.storage.db import KnowledgeBase


def _topic(kb: KnowledgeBase, topic_id: str, name: str) -> Topic:
    topic = Topic(id=topic_id, name=name, query=name, keywords=[name], sources=["local"])
    return kb.save_topic(topic)


def _paper(kb: KnowledgeBase, title: str, abstract: str = "") -> Paper:
    key = kb.upsert_paper(Paper(title=title, abstract=abstract, source="local", sources=["local"]))
    paper = kb.resolve_paper(key)
    assert paper is not None
    return paper


def test_child_papers_belong_to_the_parent_topic(tmp_path: Path):
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        parent = _topic(kb, "science", "Science of Science")
        child = _topic(kb, "promo", "Promotional Language")
        paper = _paper(kb, "Grant wording", "promotional language predicts grant success")
        kb.link_topic_paper("promo", ScoredPaper(paper, 80, None, 80, "手动", "candidate"), paper.key)
        kb.set_topic_parent(child.id, parent.id)
        listed = kb.list_topic_papers(parent.id)
        assert [item.paper.key for item in listed] == [paper.key]
        assert listed[0].inherited is True
        assert any(topic.id == parent.id for topic in kb.topics_of_paper(paper.key))
        with pytest.raises(AutoPaperError):
            kb.set_topic_parent(parent.id, child.id)


def test_shared_papers_relate_topics_and_manual_links_can_be_removed(tmp_path: Path):
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        left = _topic(kb, "alpha", "Alpha Topic")
        right = _topic(kb, "beta", "Beta Topic")
        alone = _topic(kb, "gamma", "Gamma Topic")
        paper = _paper(kb, "Shared Paper")
        kb.link_topic_paper(left.id, ScoredPaper(paper, 90, None, 90, "手动", "candidate"), paper.key)
        kb.link_topic_paper(right.id, ScoredPaper(paper, 70, None, 70, "手动", "candidate"), paper.key)
        assert ("alpha", "beta") in kb.shared_topic_pairs()
        kb.add_topic_related(left.id, alone.id)
        assert ("alpha", "gamma") in kb.manual_related_pairs()
        kb.remove_topic_related(left.id, alone.id)
        assert ("alpha", "gamma") not in kb.manual_related_pairs()
        assert ("alpha", "beta") in kb.shared_topic_pairs()


def test_suggestions_and_reclassify_use_names_in_the_text(tmp_path: Path):
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        author = kb.create_author("Huilian Qiu", ["Sophie Qiu"])
        topic = _topic(kb, "promo", "Promotional Language")
        other = _topic(kb, "climate", "Climate Coach")
        suggestion = suggest_catalog(
            "Sophie Qiu studies promotional language in grant applications.",
            [author],
            [topic, other],
        )
        assert [item["id"] for item in suggestion["authors"]] == [author.id]
        assert [item["id"] for item in suggestion["topics"]] == [topic.id]
        paper = _paper(kb, "Dashboard paper", "")
        source = tmp_path / "coach.txt"
        source.write_text("A climate coach dashboard for maintainers", encoding="utf-8")
        kb.set_local_path(paper.key, str(source))
        result = reclassify_library(kb, None)
        assert result["added"] == 1
        assert result["links"][0]["topic"] == "Climate Coach"
        assert "climate" in kb.direct_topic_ids(paper.key)


def test_each_paper_uses_its_own_author_block_abstract_and_conclusion():
    middle = "The experimental apparatus remained unchanged across trials. " * 40
    first = f"""Huilian Qiu
University of Example

Abstract
This paper studies grant wording.

1. Introduction
Ada Lovelace is cited here. quantum knitting appears only in the methods.
{middle}
5. Conclusion
Promotional language predicts success.

References
Ada Lovelace. Quantum knitting.
"""
    second = """Other Writer

Abstract
A climate coach dashboard for maintainers.

Conclusion
The dashboard stays useful.
"""
    grant = build_excerpt("grant.txt", first, "")
    coach = build_excerpt("coach.txt", second, "")
    listed = build_excerpt("listed.txt", "No author is written in the body.", "Huilian Qiu")
    assert "Huilian Qiu" in grant.author_haystack()
    assert "Ada Lovelace" not in grant.author_haystack()
    assert "promotional language" in grant.topic_haystack().casefold()
    assert "quantum knitting" not in grant.topic_haystack().casefold()
    assert "quantum knitting" not in grant.llm_text().casefold()
    assert len(grant.llm_text()) < 4000
    assert "Huilian Qiu" in listed.author_haystack()
    author = Author(id="huilian-qiu", name="Huilian Qiu", aliases=["Sophie Qiu"])
    promo = Topic(id="promo", name="Promotional Language", query="Promotional Language", keywords=["Promotional Language"], sources=["local"])
    climate = Topic(id="climate", name="Climate Coach", query="Climate Coach", keywords=["Climate Coach"], sources=["local"])
    result = suggest_many([grant, coach], [author], [promo, climate], None)
    assert [item["id"] for item in result["papers"][0]["authors"]] == ["huilian-qiu"]
    assert [item["id"] for item in result["papers"][0]["topics"]] == ["promo"]
    assert result["papers"][1]["authors"] == []
    assert [item["id"] for item in result["papers"][1]["topics"]] == ["climate"]


def test_suggestions_include_new_authors_and_paper_keywords(tmp_path: Path):
    text = """Use of Promotional Language
HuilianSophieQiu,PhD;HaoPeng,PhD;BrianUzzi,PhD
Abstract
Grant wording matters.
Corresponding Author: BrianUzzi, PhD, Kellogg School
Keywords
promotional language, grant success, science funding
"""
    excerpt = build_excerpt("grant.txt", text)
    assert excerpt.author_names[:2] == ("Huilian Sophie Qiu", "Hao Peng")
    assert excerpt.corresponding == "Brian Uzzi"
    assert excerpt.keywords[:3] == ("promotional language", "grant success", "science funding")
    suggestion = suggest_catalog(text, [], [])
    assert [item["name"] for item in suggestion["new_authors"]] == ["Huilian Sophie Qiu", "Hao Peng", "Brian Uzzi"]
    known = Author(id="huilian-qiu", name="Huilian Qiu", aliases=["Sophie Qiu"])
    matched = suggest_catalog(text, [known], [])
    assert any(item["id"] == "huilian-qiu" and item["reason"] == "一作" for item in matched["authors"])
    assert all(item["name"] != "Huilian Sophie Qiu" for item in matched["new_authors"])
    assert "、Huilian Qiu" not in matched["message"]
    assert [item["role"] for item in suggestion["new_authors"]] == ["一作", "二作", "通信作者"]
    assert [item["name"] for item in suggestion["new_topics"] if item["selected"]] == ["promotional language", "grant success"]
    assert any(item["name"] == "science funding" and item["selected"] is False for item in suggestion["new_topics"])
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        paper = _paper(kb, "Grant wording", "")
        source = tmp_path / "grant.txt"
        source.write_text(text, encoding="utf-8")
        kb.set_local_path(paper.key, str(source))
        result = reclassify_library(kb, None)
        assert result["authors_added"] == 2
        assert [item["author"] for item in result["authors"]] == ["Huilian Sophie Qiu", "Hao Peng"]
        assert [item["topic"] for item in result["links"]] == ["promotional language", "grant success"]
        linked = [author.name for author in kb.authors_of_paper(paper.key)]
        assert linked == ["Hao Peng", "Huilian Sophie Qiu"]
