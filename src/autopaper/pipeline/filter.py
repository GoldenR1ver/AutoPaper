from __future__ import annotations

from dataclasses import dataclass

from autopaper.models import Paper, Topic
from autopaper.textutil import normalize_title


@dataclass
class FilterResult:
    kept: list[Paper]
    dropped_exclude: int = 0
    dropped_keyword: int = 0
    dropped_domain: int = 0


def paper_haystack(paper: Paper) -> str:
    parts = [
        paper.title or "",
        paper.abstract or "",
        paper.venue or "",
        " ".join(paper.categories),
    ]
    return normalize_title(" ".join(parts))


def _terms(values: list[str]) -> list[str]:
    terms = []
    for value in values:
        term = normalize_title(value)
        if term:
            terms.append(term)
    return terms


def passes_keywords(topic: Topic, haystack: str) -> bool:
    keywords = _terms(topic.keywords)
    if not keywords:
        return True
    hits = [keyword in haystack for keyword in keywords]
    if topic.keyword_mode == "all":
        return all(hits)
    return any(hits)


def passes_domain(paper: Paper, topic: Topic, haystack: str) -> bool:
    tags = _terms(topic.domain_tags)
    if not tags:
        return True
    if any(tag in haystack for tag in tags):
        return True
    if topic.domain_mode == "soft" and not paper.categories:
        return True
    return False


def filter_papers(papers: list[Paper], topic: Topic) -> FilterResult:
    result = FilterResult(kept=[])
    excludes = _terms(topic.exclude_keywords)
    for paper in papers:
        haystack = paper_haystack(paper)
        if excludes and any(term in haystack for term in excludes):
            result.dropped_exclude += 1
            continue
        if not passes_keywords(topic, haystack):
            result.dropped_keyword += 1
            continue
        if not passes_domain(paper, topic, haystack):
            result.dropped_domain += 1
            continue
        result.kept.append(paper)
    return result
