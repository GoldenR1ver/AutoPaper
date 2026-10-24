from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from autopaper.errors import LLMError
from autopaper.models import Paper, Topic
from autopaper.pipeline.filter import FilterResult

log = logging.getLogger(__name__)


@dataclass
class RecallResult:
    papers: list[Paper]
    semantic: list[float | None]
    recalled: int
    warning: str | None = None


def topic_embedding_text(topic: Topic) -> str:
    parts = [topic.name, topic.query, " ".join(topic.keywords), " ".join(topic.domain_tags)]
    return "\n".join(part for part in parts if part).strip() or topic.name or "topic"


def paper_embedding_text(paper: Paper) -> str:
    abstract = (paper.abstract or "")[:1500]
    return f"{paper.title}\n{abstract}".strip()


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x_value, y_value in zip(left, right):
        dot += x_value * y_value
        left_norm += x_value * x_value
        right_norm += y_value * y_value
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def apply_recall(
    filtered: FilterResult,
    topic: Topic,
    embedder,
    *,
    min_similarity: float = 0.42,
    limit: int = 30,
    embed_limit: int = 400,
) -> RecallResult:
    """字面主题词没命中时，用主题向量把语义相近的论文补回候选。"""
    hits = list(filtered.kept)
    if embedder is None or embed_limit <= 0:
        return RecallResult(hits, [None] * len(hits), 0)

    budget = embed_limit
    chosen_hits = hits[:budget]
    budget -= len(chosen_hits)
    chosen_misses = list(filtered.keyword_misses[: max(0, budget)])
    if not chosen_hits and not chosen_misses:
        return RecallResult(hits, [None] * len(hits), 0)

    texts = [topic_embedding_text(topic)]
    texts.extend(paper_embedding_text(paper) for paper in chosen_hits)
    texts.extend(paper_embedding_text(paper) for paper in chosen_misses)
    try:
        vectors = embedder.embed(texts)
    except LLMError as exc:
        warning = f"语义召回失败，已保留关键词过滤：{exc}"
        log.warning("%s", warning)
        return RecallResult(hits, [None] * len(hits), 0, warning)
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        warning = "语义召回失败，已保留关键词过滤：向量数量和文本不一致"
        log.warning("%s", warning)
        return RecallResult(hits, [None] * len(hits), 0, warning)

    topic_vector = vectors[0]
    similarities = [cosine(topic_vector, vector) for vector in vectors[1:]]
    hit_scores: list[float | None] = [None] * len(hits)
    for offset, score in enumerate(similarities[: len(chosen_hits)]):
        hit_scores[offset] = score
    missed = list(
        zip(chosen_misses, similarities[len(chosen_hits) :], strict=False)
    )
    missed.sort(key=lambda item: item[1], reverse=True)
    picked = [(paper, score) for paper, score in missed if score >= min_similarity]
    picked = picked[: max(0, limit)]
    papers = [*hits, *[paper for paper, _score in picked]]
    semantic = [*hit_scores, *[score for _paper, score in picked]]
    return RecallResult(papers, semantic, len(picked))
