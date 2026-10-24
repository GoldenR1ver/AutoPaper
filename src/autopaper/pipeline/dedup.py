from __future__ import annotations

from autopaper.models import Paper, copy_paper, merge_papers
from autopaper.textutil import normalize_title
from rapidfuzz import fuzz


def conflicting_ids(left: Paper, right: Paper) -> bool:
    for attr in ("doi", "arxiv_id", "pubmed_id", "s2_id", "openalex_id"):
        left_value = getattr(left, attr)
        right_value = getattr(right, attr)
        if left_value and right_value and left_value != right_value:
            return True
    return False


def same_paper(left: Paper, right: Paper, threshold: float) -> bool:
    if conflicting_ids(left, right):
        return False
    for attr in ("doi", "arxiv_id", "pubmed_id", "s2_id", "openalex_id"):
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


def deduplicate(papers: list[Paper], threshold: float = 92) -> list[Paper]:
    clusters: list[Paper] = []
    for paper in papers:
        current = copy_paper(paper).cleanup()
        if not normalize_title(current.title):
            continue
        placed = False
        for index, existing in enumerate(clusters):
            if same_paper(existing, current, threshold):
                clusters[index] = merge_papers(existing, current)
                placed = True
                break
        if not placed:
            clusters.append(current)
    return clusters
