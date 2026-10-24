from __future__ import annotations

from autopaper.config import Settings
from autopaper.errors import AutoPaperError
from autopaper.http import HttpClient
from autopaper.sources.arxiv import ArxivSource
from autopaper.sources.crossref import CrossrefSource
from autopaper.sources.openalex import OpenAlexSource
from autopaper.sources.pubmed import PubMedSource
from autopaper.sources.semantic_scholar import SemanticScholarSource

# 新增数据源时：写一个带 search() 的类，把名字加到这里。
DEFAULT_SOURCES = ["arxiv", "crossref", "openalex", "pubmed", "semantic_scholar"]

ALIASES = {
    "s2": "semantic_scholar",
    "semanticscholar": "semantic_scholar",
    "semantic-scholar": "semantic_scholar",
    "alex": "openalex",
}


def canonicalize_source(name: str) -> str:
    key = name.strip().lower()
    return ALIASES.get(key, key)


def available_source_names() -> list[str]:
    return list(DEFAULT_SOURCES)


def build_sources(names: list[str], settings: Settings, http: HttpClient) -> list:
    factories = {
        "arxiv": lambda: ArxivSource(http),
        "crossref": lambda: CrossrefSource(http, settings.mailto),
        "openalex": lambda: OpenAlexSource(http, settings.mailto),
        "pubmed": lambda: PubMedSource(http, settings.polite_email, settings.ncbi_api_key),
        "semantic_scholar": lambda: SemanticScholarSource(http, settings.s2_api_key),
    }
    sources = []
    unknown = []
    for name in names:
        canonical = canonicalize_source(name)
        factory = factories.get(canonical)
        if factory is None:
            unknown.append(name)
            continue
        sources.append(factory())
    if unknown:
        known = "、".join(DEFAULT_SOURCES)
        raise AutoPaperError(f"未知数据源：{', '.join(unknown)}。可选：{known}")
    if not sources:
        raise AutoPaperError("至少选择一个数据源")
    return sources
