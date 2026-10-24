import re
from datetime import datetime, timedelta, timezone

from autopaper.errors import AmbiguousError, LLMError
from autopaper.models import Paper, ResearcherProfile, Topic, make_key
from autopaper.pipeline.dedup import block_keys, deduplicate, same_paper
from autopaper.pipeline.filter import filter_papers
from autopaper.pipeline.rank import rank_papers
from autopaper.pipeline.recall import apply_recall
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


def test_merge_fields_and_key_do_not_depend_on_order():
    arxiv = Paper(
        title="Graph Neural Networks for Time Series Forecasting",
        authors=["Zoe", "Ada"],
        venue="arXiv",
        year=2024,
        arxiv_id="2401.00001",
        source="arxiv",
    )
    published = Paper(
        title="Graph Neural Network for Time Series Forecasting",
        authors=["Ada", "Zoe"],
        venue="Nature Machine Intelligence",
        year=2023,
        doi="10.1000/stable",
        source="crossref",
    )
    forward = deduplicate([arxiv, published], 90)
    backward = deduplicate([published, arxiv], 90)
    assert len(forward) == 1 and len(backward) == 1
    assert make_key(forward[0]) == make_key(backward[0]) == "doi:10.1000/stable"
    assert forward[0].title == backward[0].title
    assert forward[0].authors == backward[0].authors
    assert forward[0].venue == backward[0].venue
    assert forward[0].year == backward[0].year
    assert forward[0].arxiv_id == "2401.00001"


def test_fuzzy_blocks_share_a_title_stem_or_doi_prefix_and_year():
    published = Paper(
        title="Science of Science with Semantics and Citations",
        doi="10.1038/s41562-024-00001",
        year=2024,
    ).cleanup()
    preprint = Paper(
        title="Science of Science with Semantic Citation Graphs",
        year=2023,
    ).cleanup()
    other = Paper(
        title="Quantum error correction beyond the surface code",
        doi="10.1103/physrev.1",
        year=2024,
    ).cleanup()
    assert set(block_keys(published)) & set(block_keys(preprint))
    assert not (set(block_keys(published)) & set(block_keys(other)))
    assert any(key.startswith("p:10.1038|y:") for key in block_keys(published))


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


def test_embedding_recall_promotes_a_synonym_into_the_llm_window():
    literal = Paper(title="Scientometrics of software", abstract="scientometrics of packages", year=2024)
    synonym = Paper(
        title="Science of science and semantics",
        abstract="how scientific fields cite each other",
        year=2024,
    )
    pasta = Paper(title="Science of cooking", abstract="pasta boiling", year=2020)
    chosen = topic(keywords=["scientometrics"], exclude_keywords=["pasta"], llm_top_n=1)
    screened = filter_papers([literal, synonym, pasta], chosen)

    class Embed:
        def embed(self, texts):
            vectors = []
            for index, text in enumerate(texts):
                if index == 0 or "cite each other" in text.casefold():
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    recalled = apply_recall(screened, chosen, Embed(), min_similarity=0.42, limit=30, embed_limit=400)
    titles = [paper.title for paper in recalled.papers]
    assert literal.title in titles
    assert synonym.title in titles
    assert pasta.title not in titles
    assert recalled.recalled == 1

    class Rerank:
        def __init__(self):
            self.user = ""

        def complete_json(self, system, user, *, expect="object", max_tokens=2048):
            self.user = user
            keys = re.findall(r"^\[(.+)\]$", user, re.MULTILINE)
            return [{"key": key, "score": 99, "reason": "语义相关", "keep": True} for key in keys]

    rerank = Rerank()
    ranked, count, warning = rank_papers(recalled.papers, chosen, rerank, recalled.semantic)
    assert warning is None and count == 1
    assert synonym.title in rerank.user
    assert literal.title not in rerank.user
    assert ranked[0].paper.title == synonym.title


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
