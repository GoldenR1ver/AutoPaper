from datetime import datetime, timedelta, timezone

from autopaper.errors import AmbiguousError, LLMError
from autopaper.models import Paper, ResearcherProfile, Topic
from autopaper.pipeline.dedup import deduplicate, same_paper
from autopaper.pipeline.filter import filter_papers
from autopaper.pipeline.rank import rank_papers
from autopaper.scheduler import is_due
import pytest

from autopaper.analysis.reader import render_reading
from autopaper.analysis.researcher import choose_profile


def topic(**kwargs) -> Topic:
    base = dict(id="gnn", name="GNN", query="graph neural networks", sources=["arxiv"])
    base.update(kwargs)
    return Topic(**base)


def test_dedup_merges_doi_hyphen_and_near_titles_but_not_different_dois():
    first = Paper(
        title="Graph Neural Networks for Time Series Forecasting",
        doi="https://doi.org/10.1000/AAA",
        abstract="short",
        source="arxiv",
        arxiv_id="2401.00001",
    )
    second = Paper(
        title="Graph Neural Networks for Time-Series Forecasting",
        doi="10.1000/aaa",
        abstract="a much longer abstract about forecasting",
        citation_count=5,
        source="crossref",
    )
    near = Paper(title="Graph Neural Network for Time Series Forecasting", year=2023, source="openalex")
    other = Paper(title="Graph Neural Networks for Time Series Forecasting", doi="10.1000/bbb", source="pubmed")
    editorial = Paper(title="Editorial", source="crossref")
    editorial_2 = Paper(title="Editorials", source="crossref")
    merged = deduplicate([first, second, near, other, editorial, editorial_2], 90)
    by_doi = {paper.doi: paper for paper in merged if paper.doi}
    assert by_doi["10.1000/aaa"].abstract.startswith("a much longer")
    assert by_doi["10.1000/aaa"].arxiv_id == "2401.00001"
    assert by_doi["10.1000/aaa"].citation_count == 5
    assert "10.1000/bbb" in by_doi
    assert sum(paper.title.startswith("Editorial") for paper in merged) == 2
    assert same_paper(first, near, 90)


def test_keyword_exclude_and_domain_filters():
    kept_source = [
        Paper(title="Time-series graph neural networks", abstract="forecasting", categories=["cs.LG"]),
        Paper(title="A survey of graph neural networks", abstract="time series", categories=["cs.LG"]),
        Paper(title="Protein folding", abstract="biology", categories=["q-bio"]),
        Paper(title="Graph models without a category", abstract="time series"),
    ]
    result = filter_papers(
        kept_source,
        topic(
            keywords=["time series"],
            exclude_keywords=["survey"],
            domain_tags=["cs.LG"],
            domain_mode="strict",
            keyword_mode="any",
        ),
    )
    assert [paper.title for paper in result.kept] == ["Time-series graph neural networks"]
    assert result.dropped_exclude == 1
    assert result.dropped_domain == 1
    soft = filter_papers(
        kept_source,
        topic(keywords=["time series"], domain_tags=["cs.LG"], domain_mode="soft"),
    )
    assert any(paper.title.startswith("Graph models") for paper in soft.kept)
    both = filter_papers(
        [Paper(title="Graph neural networks", abstract="for time series")],
        topic(keywords=["graph", "protein"], keyword_mode="all"),
    )
    assert both.kept == []


def test_rank_uses_keywords_then_llm_can_reorder():
    current = datetime.now(timezone.utc).year
    relevant = Paper(title="Graph neural networks", abstract="time series forecasting", year=current)
    other = Paper(title="Cooking pasta", abstract="boiling water", year=2000)
    ranked, count, warning = rank_papers([other, relevant], topic(keywords=["graph neural networks"]), None)
    assert warning is None and count == 0
    assert ranked[0].paper.title.startswith("Graph")

    class FakeLLM:
        def complete_json(self, system, user, *, expect="object", max_tokens=2048):
            import re

            keys = re.findall(r"^\[(.+)\]$", user, re.MULTILINE)
            return [
                {"key": key, "score": 10 if index == 0 else 99, "reason": "测试理由", "keep": True}
                for index, key in enumerate(keys)
            ]

    reranked, count, warning = rank_papers(
        [relevant, other],
        topic(keywords=["graph neural networks"], llm_top_n=5),
        FakeLLM(),
    )
    assert count == 2 and warning is None
    assert reranked[0].paper.title == "Cooking pasta"
    assert reranked[0].llm_score == 99

    class Boom:
        def complete_json(self, *args, **kwargs):
            raise LLMError("down")

    fallback, count, warning = rank_papers([relevant, other], topic(llm_top_n=5), Boom())
    assert count == 0 and warning
    assert fallback[0].paper.title.startswith("Graph")


def test_schedule_windows_and_manual_topics_never_run_themselves():
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    daily = topic(schedule="daily", last_run_at=now - timedelta(hours=21))
    assert is_due(daily, now)
    daily.last_run_at = now - timedelta(hours=3)
    assert not is_due(daily, now)
    weekly = topic(schedule="weekly", last_run_at=now - timedelta(days=7))
    assert is_due(weekly, now)
    assert not is_due(topic(schedule="manual"), now)
    assert is_due(topic(schedule="daily", last_run_at=None), now)


def test_reading_template_contains_the_requested_questions():
    paper = Paper(title="A paper", authors=["Ada"], year=2024, doi="10.1/a")
    markdown = render_reading(
        paper,
        {
            "field": "时间序列",
            "innovations": ["模型", "训练方式"],
            "reproduction": "先准备数据，再复现训练。",
        },
    )
    for heading in ("所在领域", "知识网络中的坐标", "能否解决该问题", "创新点", "复现思路"):
        assert heading in markdown
    assert "- 模型" in markdown
    structured = render_reading(
        paper,
        {
            "innovations": {"method": ["新指标"], "direction": ["科学学"], "theory": []},
            "reproduction": {"data": "公开申请书", "steps": ["标注", "回归"], "risks": "样本偏差"},
        },
    )
    assert "### 方法" in structured
    assert "- 新指标" in structured
    assert "数据与材料：公开申请书" in structured


def test_ambiguous_researcher_asks_for_pick():
    people = [
        ResearcherProfile(name="Jane Doe", affiliation="MIT", works_count=20, openalex_id="A1"),
        ResearcherProfile(name="Jane Doe", affiliation="Stanford", works_count=19, openalex_id="A2"),
    ]
    with pytest.raises(AmbiguousError) as caught:
        choose_profile("Jane Doe", people, None)
    assert "--pick" in str(caught.value)
    chosen = choose_profile("Jane Doe", people, 2)
    assert chosen.openalex_id == "A2"
