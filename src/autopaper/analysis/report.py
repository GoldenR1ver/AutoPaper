from __future__ import annotations

from collections import Counter
from pathlib import Path

from autopaper.errors import LLMError
from autopaper.llm.prompts import REPORT_SYSTEM, report_user_prompt
from autopaper.models import ScoredPaper, Topic
from autopaper.storage.db import KnowledgeBase
from autopaper.storage.export import reference_list


def _top(counter: Counter, limit: int) -> list[tuple[str, int]]:
    return [(name, count) for name, count in counter.most_common(limit) if name]


def render_topic_report(topic: Topic, items: list[ScoredPaper], run: dict | None, narrative: str | None) -> str:
    candidates = [item for item in items if item.status == "candidate"]
    shown = candidates or items
    lines = [
        f"# {topic.name} 研究报告",
        "",
        f"- 检索式：{topic.query}",
        f"- 主题词：{'、'.join(topic.keywords) or '无'}",
        f"- 排除词：{'、'.join(topic.exclude_keywords) or '无'}",
        f"- 领域标签：{'、'.join(topic.domain_tags) or '无'}（{topic.domain_mode}）",
        f"- 数据源：{'、'.join(topic.sources)}",
        f"- 起始年份：{topic.since_year or '不限'}",
        "",
        "## 检索情况",
        "",
    ]
    if run is None:
        lines.append("还没有检索记录。")
    else:
        stats = run.get("stats") or {}
        fetched = stats.get("fetched") or {}
        fetched_text = "、".join(f"{name} {count} 篇" for name, count in fetched.items()) or "无"
        lines.append(f"- 时间：{run.get('finished_at') or run.get('started_at')}")
        lines.append(f"- 状态：{run.get('status')}")
        lines.append(f"- 各源结果：{fetched_text}")
        lines.append(
            f"- 去重后 {stats.get('deduped_count', 0)} 篇，过滤后 {stats.get('filtered_count', 0)} 篇，"
            f"候选 {stats.get('candidate_count', 0)} 篇"
        )
        errors = stats.get("errors") or []
        if errors:
            lines.append("- 问题：" + "；".join(str(item) for item in errors))
    lines.extend(["", "## 文献概览", ""])
    if not shown:
        lines.append("知识库里还没有这个主题的论文。")
    else:
        years = [item.paper.year for item in shown if item.paper.year]
        year_text = f"{min(years)}–{max(years)}" if years else "未知"
        lines.append(f"当前纳入 {len(shown)} 篇，年份范围 {year_text}。")
        lines.append("")
        by_year = Counter(item.paper.year for item in shown if item.paper.year)
        if by_year:
            lines.append("年份分布：")
            for year, count in sorted(by_year.items()):
                lines.append(f"- {year}：{count} 篇")
            lines.append("")
        venues = Counter(item.paper.venue for item in shown if item.paper.venue)
        if venues:
            lines.append("出现较多的期刊或会议：")
            for name, count in _top(venues, 8):
                lines.append(f"- {name}：{count} 篇")
            lines.append("")
        authors = Counter(author for item in shown for author in item.paper.authors)
        if authors:
            lines.append("出现较多的作者：")
            for name, count in _top(authors, 8):
                lines.append(f"- {name}：{count} 篇")
            lines.append("")
    lines.extend(["## 代表论文", ""])
    for item in shown[:10]:
        paper = item.paper
        lines.append(f"### {paper.title}")
        lines.append("")
        lines.append(
            f"{paper.year or '年份未知'}，{paper.venue or '出处未知'}，"
            f"被引 {paper.citation_count or 0}，相关性 {item.score:.1f}。"
        )
        lines.append(item.reason)
        abstract = (paper.abstract or "（无摘要）")[:300]
        lines.append("")
        lines.append(abstract)
        lines.append("")
    lines.extend(["## 领域梳理", ""])
    if narrative:
        lines.append(narrative.strip())
    else:
        lines.append("未生成模型综述。配置 OPENAI_API_KEY 后运行 `autopaper report 主题` 可以补上这一节。")
    lines.extend(["", "## 引用列表", "", reference_list([item.paper for item in shown[:100]]), ""])
    if len(shown) > 100:
        lines.append(f"引用列表只列出前 100 篇，知识库中还有 {len(shown) - 100} 篇。")
        lines.append("")
    return "\n".join(lines)


def _evidence(items: list[ScoredPaper]) -> str:
    blocks = []
    for item in items[:12]:
        paper = item.paper
        abstract = (paper.abstract or "（无摘要）")[:700]
        blocks.append(
            f"标题：{paper.title}\n年份：{paper.year or '未知'}\n出处：{paper.venue or '未知'}\n摘要：{abstract}"
        )
    return "\n\n".join(blocks)


def write_topic_report(kb: KnowledgeBase, topic: Topic, settings, llm=None) -> Path:
    items = kb.list_topic_papers(topic.id, include_held=False)
    if not items:
        items = kb.list_topic_papers(topic.id, include_held=True)
    narrative = None
    if llm is not None and items:
        try:
            narrative = llm.complete(
                REPORT_SYSTEM,
                report_user_prompt(topic, _evidence(items)),
                max_tokens=2500,
            )
        except LLMError as exc:
            narrative = f"模型综述生成失败：{exc}\n\n下面的题录和统计仍然来自本地知识库。"
    markdown = render_topic_report(topic, items, kb.latest_run(topic.id), narrative)
    path = settings.data_dir / "reports" / "topics" / f"{topic.id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
