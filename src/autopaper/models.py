from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime

from autopaper.textutil import collapse_ws, dedupe_names, max_optional, normalize_title


@dataclass
class Paper:
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pubmed_id: str | None = None
    s2_id: str | None = None
    openalex_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    categories: list[str] = field(default_factory=list)
    source: str = ""
    sources: list[str] = field(default_factory=list)
    key: str = ""
    local_path: str = ""

    def cleanup(self) -> Paper:
        from autopaper.textutil import (
            normalize_arxiv_id,
            normalize_doi,
            normalize_pmid,
            openalex_short_id,
        )

        self.title = collapse_ws(self.title)
        self.abstract = collapse_ws(self.abstract)[:20000]
        self.year = _as_int(self.year)
        self.citation_count = _as_int(self.citation_count)
        self.doi = normalize_doi(self.doi)
        self.arxiv_id = normalize_arxiv_id(self.arxiv_id)
        self.pubmed_id = normalize_pmid(self.pubmed_id)
        self.openalex_id = openalex_short_id(self.openalex_id)
        self.s2_id = collapse_ws(self.s2_id) or None
        self.url = collapse_ws(self.url) or None
        self.pdf_url = collapse_ws(self.pdf_url) or None
        self.venue = collapse_ws(self.venue) or None
        self.authors = dedupe_names(self.authors)
        self.categories = dedupe_names(self.categories)
        names = dedupe_names([*self.sources, self.source])
        self.sources = names
        if self.source and self.source not in self.sources:
            self.sources.insert(0, self.source)
        if not self.source and self.sources:
            self.source = self.sources[0]
        return self


@dataclass
class Topic:
    id: str
    name: str
    query: str
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    domain_mode: str = "soft"
    sources: list[str] = field(default_factory=list)
    schedule: str = "manual"
    sort: str = "relevance"
    max_per_source: int = 20
    since_year: int | None = None
    min_score: float = 0
    llm_top_n: int = 20
    keyword_mode: str = "any"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_run_at: datetime | None = None


@dataclass
class ScoredPaper:
    paper: Paper
    heuristic_score: float
    llm_score: float | None
    score: float
    reason: str
    status: str


@dataclass
class Author:
    id: str
    name: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class ResearcherProfile:
    name: str
    orcid: str | None = None
    affiliation: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    h_index: int | None = None
    cited_by_count: int | None = None
    works_count: int | None = None
    concepts: list[str] = field(default_factory=list)
    topic_ids: list[str] = field(default_factory=list)
    match_score: float = 0


@dataclass
class Relation:
    other_name: str
    relation: str
    weight: float
    detail: str = ""
    other_id: str | None = None


def _as_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def copy_paper(paper: Paper) -> Paper:
    return replace(
        paper,
        authors=list(paper.authors),
        categories=list(paper.categories),
        sources=list(paper.sources),
    )


def make_key(paper: Paper) -> str:
    if paper.doi:
        return f"doi:{paper.doi}"
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id}"
    if paper.pubmed_id:
        return f"pmid:{paper.pubmed_id}"
    if paper.openalex_id:
        return f"oa:{paper.openalex_id}"
    if paper.s2_id:
        return f"s2:{paper.s2_id}"
    digest = hashlib.sha1(normalize_title(paper.title).encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"


def prefer_year(left: Paper, right: Paper) -> int | None:
    if left.year and right.year and left.year != right.year:
        if left.doi and not right.doi:
            return left.year
        if right.doi and not left.doi:
            return right.year
        return max(left.year, right.year)
    return left.year or right.year


def merge_papers(left: Paper, right: Paper) -> Paper:
    """合并同一篇论文的两条记录，补齐缺失字段。"""
    left_title = left.title or ""
    right_title = right.title or ""
    title = left_title if len(left_title) >= len(right_title) else right_title
    left_abstract = left.abstract or ""
    right_abstract = right.abstract or ""
    abstract = left_abstract if len(left_abstract) >= len(right_abstract) else right_abstract
    merged = Paper(
        title=title,
        abstract=abstract,
        authors=dedupe_names([*left.authors, *right.authors]),
        year=prefer_year(left, right),
        doi=left.doi or right.doi,
        arxiv_id=left.arxiv_id or right.arxiv_id,
        pubmed_id=left.pubmed_id or right.pubmed_id,
        s2_id=left.s2_id or right.s2_id,
        openalex_id=left.openalex_id or right.openalex_id,
        url=left.url or right.url,
        pdf_url=left.pdf_url or right.pdf_url,
        venue=left.venue or right.venue,
        citation_count=max_optional(left.citation_count, right.citation_count),
        categories=dedupe_names([*left.categories, *right.categories]),
        source=left.source or right.source,
        sources=dedupe_names([*left.sources, *right.sources, left.source, right.source]),
    )
    return merged.cleanup()
