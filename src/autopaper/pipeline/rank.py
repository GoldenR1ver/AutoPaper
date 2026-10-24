from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from autopaper.errors import LLMError
from autopaper.llm.prompts import RANK_SYSTEM, rank_user_prompt
from autopaper.models import Paper, ScoredPaper, Topic, make_key
from autopaper.pipeline.filter import paper_haystack
from autopaper.textutil import normalize_title

log = logging.getLogger(__name__)


def topic_terms(topic: Topic) -> list[str]:
    keywords = [normalize_title(item) for item in topic.keywords if normalize_title(item)]
    if keywords:
        return keywords
    query = normalize_title(topic.query or topic.name)
    tokens = [token for token in query.split() if len(token) >= 3]
    return tokens or ([query] if query else [])


def heuristic_score(paper: Paper, topic: Topic) -> tuple[float, str]:
    terms = topic_terms(topic)
    haystack = paper_haystack(paper)
    title = normalize_title(paper.title)
    if terms:
        coverage = sum(term in haystack for term in terms) / len(terms)
        title_hits = sum(term in title for term in terms)
        title_score = title_hits / len(terms)
    else:
        coverage = 0.5
        title_hits = 0
        title_score = 0.0
    citations = paper.citation_count or 0
    cite_score = min(1.0, math.log1p(citations) / math.log1p(500))
    if paper.year:
        age = max(0, datetime.now(timezone.utc).year - paper.year)
        recency = max(0.0, 1.0 - age / 15)
    else:
        recency = 0.3
    score = 100 * (0.50 * coverage + 0.20 * title_score + 0.15 * cite_score + 0.15 * recency)
    tags = [normalize_title(tag) for tag in topic.domain_tags if normalize_title(tag)]
    if tags and any(tag in haystack for tag in tags):
        score += 5
    score = round(min(100.0, max(0.0, score)), 1)
    reason = (
        f"关键词覆盖 {coverage:.0%}，标题命中 {title_hits}/{len(terms)}，"
        f"被引 {citations}，年份 {paper.year or '未知'}"
    )
    return score, reason


def _status(score: float, topic: Topic) -> str:
    return "candidate" if score >= topic.min_score else "held"


def _coerce_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, score))


def rank_papers(papers: list[Paper], topic: Topic, llm=None) -> tuple[list[ScoredPaper], int, str | None]:
    scored: list[ScoredPaper] = []
    for paper in papers:
        value, reason = heuristic_score(paper, topic)
        scored.append(
            ScoredPaper(
                paper=paper,
                heuristic_score=value,
                llm_score=None,
                score=value,
                reason=reason,
                status=_status(value, topic),
            )
        )
    scored.sort(key=lambda item: (item.score, item.paper.citation_count or 0, item.paper.year or 0), reverse=True)
    warning = None
    llm_count = 0
    if llm is None or topic.llm_top_n <= 0 or not scored:
        return scored, llm_count, warning

    chosen = scored[: topic.llm_top_n]
    blocks = []
    by_key: dict[str, ScoredPaper] = {}
    for item in chosen:
        key = make_key(item.paper)
        by_key[key] = item
        abstract = (item.paper.abstract or "（无摘要）")[:1000]
        blocks.append(f"[{key}]\n标题：{item.paper.title}\n摘要：{abstract}")
    try:
        payload = llm.complete_json(
            RANK_SYSTEM,
            rank_user_prompt(topic, "\n\n".join(blocks)),
            expect="array",
            max_tokens=2200,
        )
    except LLMError as exc:
        warning = f"模型打分失败，已保留启发式排序：{exc}"
        log.warning(warning)
        return scored, 0, warning

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "")
        item = by_key.get(key)
        score = _coerce_score(entry.get("score"))
        if item is None or score is None:
            continue
        reason = str(entry.get("reason") or "").strip()[:200] or item.reason
        if entry.get("keep") is False:
            reason = f"模型建议跳过。{reason}"
        item.llm_score = score
        item.score = score
        item.reason = reason
        item.status = _status(score, topic)
        llm_count += 1
    scored.sort(key=lambda item: (item.score, item.paper.citation_count or 0, item.paper.year or 0), reverse=True)
    return scored, llm_count, warning
