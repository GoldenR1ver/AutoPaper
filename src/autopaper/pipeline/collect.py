from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from autopaper.analysis.report import write_topic_report
from autopaper.errors import SourceError
from autopaper.models import Topic
from autopaper.llm.client import maybe_embedder
from autopaper.pipeline.dedup import deduplicate
from autopaper.pipeline.filter import filter_papers
from autopaper.pipeline.rank import rank_papers
from autopaper.pipeline.recall import apply_recall
from autopaper.storage.db import KnowledgeBase
from autopaper.storage.export import export_topic_bundle

log = logging.getLogger(__name__)


@dataclass
class CollectStats:
    topic_id: str
    status: str
    fetched: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    raw_count: int = 0
    deduped_count: int = 0
    filtered_count: int = 0
    dropped_exclude: int = 0
    dropped_keyword: int = 0
    dropped_domain: int = 0
    stored_count: int = 0
    candidate_count: int = 0
    recalled_count: int = 0
    llm_scored: int = 0
    export_dir: str = ""
    report_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _embedder_for(settings, llm):
    if llm is not None and callable(getattr(llm, "embed", None)):
        return llm, False
    if getattr(settings, "disable_llm", False):
        return None, False
    embedder = maybe_embedder(settings)
    return embedder, embedder is not None


def format_stats(stats: CollectStats) -> str:
    fetched = "、".join(f"{name} {count} 篇" for name, count in stats.fetched.items()) or "无"
    lines = [
        f"状态：{stats.status}",
        f"检索：{fetched}",
        (
            f"合计 {stats.raw_count} 篇，去重后 {stats.deduped_count} 篇，"
            f"过滤后 {stats.filtered_count} 篇（语义召回 {stats.recalled_count} 篇）"
        ),
        (
            f"过滤明细：排除词 {stats.dropped_exclude}，主题词未命中 {stats.dropped_keyword}，"
            f"领域标签未命中 {stats.dropped_domain}"
        ),
        f"写入 {stats.stored_count} 篇，候选 {stats.candidate_count} 篇，模型打分 {stats.llm_scored} 篇",
    ]
    if stats.export_dir:
        lines.append(f"导出：{stats.export_dir}")
    if stats.report_path:
        lines.append(f"报告：{stats.report_path}")
    if stats.errors:
        lines.append("问题：" + "；".join(stats.errors))
    return "\n".join(lines)


def run_collect(
    topic: Topic,
    kb: KnowledgeBase,
    settings,
    *,
    sources,
    llm=None,
    fresh: bool = False,
    export: bool = True,
    with_report: bool = False,
) -> CollectStats:
    run_id = kb.start_run(topic.id)
    errors: list[str] = []
    fetched: dict[str, int] = {}
    raw = []
    any_success = False
    limit = min(max(topic.max_per_source, 1), 200)
    for source in sources:
        try:
            found = source.search(
                topic.query,
                limit=limit,
                since_year=topic.since_year,
                sort=topic.sort,
            )
        except SourceError as exc:
            fetched[source.name] = 0
            errors.append(str(exc))
            log.warning("%s", exc)
            continue
        except Exception as exc:
            fetched[source.name] = 0
            errors.append(f"{source.name}: {exc}")
            log.exception("数据源 %s 失败", source.name)
            continue
        fetched[source.name] = len(found)
        raw.extend(found)
        any_success = True
        log.info("%s 返回 %s 篇", source.name, len(found))

    deduped = deduplicate(raw, settings.similarity_threshold)
    filtered = filter_papers(deduped, topic)
    embedder, owned_embedder = _embedder_for(settings, llm)
    try:
        recalled = apply_recall(
            filtered,
            topic,
            embedder,
            min_similarity=float(getattr(settings, "recall_similarity", 0.42)),
            limit=int(getattr(settings, "recall_limit", 30)),
            embed_limit=int(getattr(settings, "recall_embed_limit", 400)),
        )
    finally:
        if owned_embedder and embedder is not None:
            embedder.close()
    if recalled.warning:
        errors.append(recalled.warning)
    ranked, llm_count, warning = rank_papers(recalled.papers, topic, llm, recalled.semantic)
    if warning:
        errors.append(warning)
    if fresh:
        kb.clear_topic_papers(topic.id)
    stored = 0
    candidates = 0
    for item in ranked:
        try:
            key = kb.upsert_paper(item.paper)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        item.paper.key = key
        kb.link_topic_paper(topic.id, item, key)
        stored += 1
        if item.status == "candidate":
            candidates += 1

    if not any_success and sources:
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "success"
    if any_success:
        kb.mark_topic_ran(topic.id)

    export_dir = ""
    report_path = ""
    if export and any_success:
        try:
            items = kb.list_topic_papers(topic.id, include_held=False)
            latest = export_topic_bundle(settings.data_dir, topic.id, topic.name, items)
            export_dir = str(latest)
        except Exception as exc:
            errors.append(f"导出失败: {exc}")
            if status == "success":
                status = "partial"
    if with_report and any_success:
        try:
            report_path = str(write_topic_report(kb, topic, settings, llm))
        except Exception as exc:
            errors.append(f"报告失败: {exc}")
            if status == "success":
                status = "partial"

    stats = CollectStats(
        topic_id=topic.id,
        status=status,
        fetched=fetched,
        errors=errors,
        raw_count=len(raw),
        deduped_count=len(deduped),
        filtered_count=len(recalled.papers),
        dropped_exclude=filtered.dropped_exclude,
        dropped_keyword=max(0, filtered.dropped_keyword - recalled.recalled),
        dropped_domain=filtered.dropped_domain,
        stored_count=stored,
        candidate_count=candidates,
        recalled_count=recalled.recalled,
        llm_scored=llm_count,
        export_dir=export_dir,
        report_path=report_path,
    )
    kb.finish_run(run_id, status, stats.to_dict())
    return stats
