from __future__ import annotations

from collections import defaultdict

from autopaper.models import Paper, copy_paper, make_key, merge_papers
from autopaper.textutil import normalize_title
from rapidfuzz import fuzz

_ID_FIELDS = ("doi", "arxiv_id", "pubmed_id", "s2_id", "openalex_id")


def doi_prefix(doi: str | None) -> str:
    """DOI 注册前缀，例如 10.1038/s41562-024-1 的 10.1038。"""
    if not doi or "/" not in doi:
        return ""
    return doi.split("/", 1)[0]


def block_keys(paper: Paper) -> list[str]:
    """模糊比较只用这些桶里的候选，避免和全部已见论文逐一比对。

    桶有两种：DOI 前缀加年份（相邻年份各算一桶，预印本和正式版差一年也能碰上），
    以及规范化标题的前 18 个字符。后者用来接住没有同一 DOI 前缀的同一篇论文。
    """
    keys: list[str] = []
    title = normalize_title(paper.title)
    if len(title) >= 24:
        keys.append(f"s:{title[:18]}")
    prefix = doi_prefix(paper.doi)
    if prefix:
        if paper.year:
            for year in (paper.year - 1, paper.year, paper.year + 1):
                keys.append(f"p:{prefix}|y:{year}")
        else:
            keys.append(f"p:{prefix}|y:?")
    return keys


def conflicting_ids(left: Paper, right: Paper) -> bool:
    for attr in _ID_FIELDS:
        left_value = getattr(left, attr)
        right_value = getattr(right, attr)
        if left_value and right_value and left_value != right_value:
            return True
    return False


def same_paper(left: Paper, right: Paper, threshold: float) -> bool:
    if conflicting_ids(left, right):
        return False
    for attr in _ID_FIELDS:
        left_value = getattr(left, attr)
        right_value = getattr(right, attr)
        if left_value and right_value and left_value == right_value:
            return True
    left_title = normalize_title(left.title)
    right_title = normalize_title(right.title)
    if not left_title or not right_title:
        return False
    if left_title == right_title:
        return True
    if min(len(left_title), len(right_title)) < 24:
        return False
    return fuzz.ratio(left_title, right_title) >= threshold


class _Clusters:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.clusters: list[Paper] = []
        self.ids: dict[tuple[str, str], int] = {}
        self.titles: dict[str, list[int]] = defaultdict(list)
        self.blocks: dict[str, list[int]] = defaultdict(list)

    def add(self, paper: Paper) -> None:
        current = copy_paper(paper).cleanup()
        if not normalize_title(current.title):
            return
        match = self._find(current)
        if match is None:
            current.key = make_key(current)
            index = len(self.clusters)
            self.clusters.append(current)
            self._remember(index, current)
            return
        merged = merge_papers(self.clusters[match], current)
        self.clusters[match] = merged
        self._remember(match, current)
        self._remember(match, merged)

    def _find(self, current: Paper) -> int | None:
        seen: set[int] = set()
        for index in self._candidates(current):
            if index in seen:
                continue
            seen.add(index)
            if same_paper(self.clusters[index], current, self.threshold):
                return index
        return None

    def _candidates(self, paper: Paper):
        for attr in _ID_FIELDS:
            value = getattr(paper, attr)
            if value and (attr, value) in self.ids:
                yield self.ids[(attr, value)]
        title = normalize_title(paper.title)
        if title:
            yield from self.titles.get(title, ())
        for key in block_keys(paper):
            yield from self.blocks.get(key, ())

    def _remember(self, index: int, paper: Paper) -> None:
        for attr in _ID_FIELDS:
            value = getattr(paper, attr)
            if value:
                self.ids[(attr, value)] = index
        title = normalize_title(paper.title)
        if title:
            bucket = self.titles[title]
            if not bucket or bucket[-1] != index:
                bucket.append(index)
        for key in block_keys(paper):
            bucket = self.blocks[key]
            if not bucket or bucket[-1] != index:
                bucket.append(index)


def deduplicate(papers: list[Paper], threshold: float = 92) -> list[Paper]:
    clusters = _Clusters(threshold)
    for paper in papers:
        clusters.add(paper)
    return clusters.clusters
