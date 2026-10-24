from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from autopaper.errors import AutoPaperError, LLMError, NotFoundError, SourceError, AmbiguousError
from autopaper.llm.prompts import RESEARCHER_SYSTEM, researcher_user_prompt
from autopaper.models import Paper, Relation, ResearcherProfile
from autopaper.pipeline.dedup import deduplicate
from autopaper.storage.db import KnowledgeBase
from autopaper.storage.export import format_reference
from autopaper.textutil import normalize_match, slugify


def researcher_storage_id(profile: ResearcherProfile) -> str:
    base = slugify(profile.name)
    if profile.openalex_id:
        return f"{base}-{profile.openalex_id.lower()}"
    if profile.s2_id:
        return f"{base}-{profile.s2_id[:12]}"
    return base


def rank_profiles(query: str, profiles: list[ResearcherProfile]) -> list[ResearcherProfile]:
    for profile in profiles:
        profile.match_score = float(fuzz.token_set_ratio(query, profile.name))
    return sorted(profiles, key=lambda item: (item.match_score, item.cited_by_count or 0), reverse=True)


def format_candidates(profiles: list[ResearcherProfile]) -> str:
    lines = ["找到多位可能的研究者，请用 --pick 指定序号："]
    for index, profile in enumerate(profiles, start=1):
        lines.append(
            f"{index}. {profile.name} | {profile.affiliation or '机构未知'} | "
            f"论文 {profile.works_count or 0} | h-index {profile.h_index or '未知'} | "
            f"{profile.openalex_id or profile.s2_id or '无编号'} | 匹配 {profile.match_score:.0f}"
        )
    return "\n".join(lines)


def choose_profile(query: str, profiles: list[ResearcherProfile], pick: int | None) -> ResearcherProfile:
    ranked = rank_profiles(query, profiles)
    if not ranked:
        raise NotFoundError(f"没有找到研究者：{query}")
    if pick is not None:
        if pick < 1 or pick > len(ranked):
            raise AutoPaperError(f"--pick 需要在 1 到 {len(ranked)} 之间")
        return ranked[pick - 1]
    good = [item for item in ranked if item.match_score >= 70]
    if len(good) >= 2 and good[0].match_score - good[1].match_score < 8:
        raise AmbiguousError(format_candidates(ranked[:8]))
    return good[0] if good else ranked[0]


def _coauthors(profile: ResearcherProfile, papers: list[Paper]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    own = normalize_match(profile.name)
    for paper in papers:
        for author in paper.authors:
            if normalize_match(author) == own:
                continue
            counter[author] += 1
    pairs = counter.most_common(15)
    if len(papers) >= 8:
        repeated = [item for item in pairs if item[1] >= 2]
        return repeated or pairs[:8]
    return pairs


def _fallback_summary(profile: ResearcherProfile, papers: list[Paper]) -> tuple[str, list[str]]:
    concepts = "、".join(profile.concepts[:6]) or "尚未标注"
    years = [paper.year for paper in papers if paper.year]
    span = f"{min(years)}–{max(years)}" if years else "年份未知"
    summary = (
        f"{profile.name} 的公开记录约 {profile.works_count or len(papers)} 篇论文，"
        f"被引 {profile.cited_by_count or 0}，h-index {profile.h_index if profile.h_index is not None else '未知'}。"
        f"主要主题包括 {concepts}。本次采样 {len(papers)} 篇，时间跨度 {span}。"
    )
    return summary, profile.concepts[:6]


def _mermaid(profile: ResearcherProfile, relations: list[Relation]) -> str:
    lines = ["```mermaid", "graph LR"]
    labels = {"self": profile.name}
    for index, relation in enumerate(relations[:12], start=1):
        labels[f"n{index}"] = relation.other_name
    for node_id, label in labels.items():
        safe = label.replace('"', "'")
        lines.append(f'  {node_id}["{safe}"]')
    for index, relation in enumerate(relations[:12], start=1):
        label = "合作" if relation.relation == "coauthor" else "同方向"
        lines.append(f"  self -->|{label} {int(relation.weight)}| n{index}")
    lines.append("```")
    return "\n".join(lines)


def render_researcher(
    profile: ResearcherProfile,
    papers: list[Paper],
    relations: list[Relation],
    summary: str,
    directions: list[str],
    relations_note: str,
) -> str:
    influential = sorted(papers, key=lambda item: item.citation_count or 0, reverse=True)[:8]
    recent = sorted(papers, key=lambda item: item.year or 0, reverse=True)[:8]
    lines = [
        f"# {profile.name}",
        "",
        f"- 机构：{profile.affiliation or '未知'}",
        f"- ORCID：{profile.orcid or '无'}",
        f"- OpenAlex：{profile.openalex_id or '无'}",
        f"- Semantic Scholar：{profile.s2_id or '无'}",
        f"- 论文数：{profile.works_count if profile.works_count is not None else '未知'}",
        f"- 被引：{profile.cited_by_count if profile.cited_by_count is not None else '未知'}",
        f"- h-index：{profile.h_index if profile.h_index is not None else '未知'}",
        "",
        "## 工作内容与方向",
        "",
        summary.strip(),
        "",
    ]
    if directions:
        for direction in directions:
            lines.append(f"- {direction}")
        lines.append("")
    if profile.match_score and profile.match_score < 80:
        lines.append(f"名字匹配度是 {profile.match_score:.0f}。如果这不是目标研究者，请用 --pick 或 --openalex-id 重新指定。")
        lines.append("")
    lines.extend(["## 高被引论文", ""])
    if influential:
        lines.append("\n".join(format_reference(index, paper) for index, paper in enumerate(influential, start=1)))
    else:
        lines.append("没有取到论文题录。")
    lines.extend(["", "## 近年论文", ""])
    if recent:
        lines.append("\n".join(format_reference(index, paper) for index, paper in enumerate(recent, start=1)))
    else:
        lines.append("没有取到论文题录。")
    lines.extend(["", "## 合作者", ""])
    coauthors = [item for item in relations if item.relation == "coauthor"]
    if coauthors:
        for item in coauthors:
            lines.append(f"- {item.other_name}：共同出现 {int(item.weight)} 篇")
    else:
        lines.append("采样论文里没有足够的合作者信息。")
    lines.extend(["", "## 同方向研究者", ""])
    peers = [item for item in relations if item.relation == "same_topic"]
    if peers:
        for item in peers:
            lines.append(f"- {item.other_name}：{item.detail or '主题相近'}")
    else:
        lines.append("没有取到同方向研究者。")
    lines.extend(["", "## 关系说明", ""])
    lines.append(relations_note.strip() or "合作关系来自本次采样的论文，不是完整的职业生涯图谱。同方向研究者来自 OpenAlex 的主题邻居。")
    lines.extend(["", "## 关系图", "", _mermaid(profile, relations), ""])
    return "\n".join(lines)


def _graph(profile: ResearcherProfile, relations: list[Relation]) -> dict:
    nodes = [{"id": "self", "label": profile.name, "type": "researcher"}]
    edges = []
    for index, relation in enumerate(relations, start=1):
        node_id = f"n{index}"
        nodes.append({"id": node_id, "label": relation.other_name, "type": relation.relation})
        edges.append(
            {
                "source": "self",
                "target": node_id,
                "relation": relation.relation,
                "weight": relation.weight,
            }
        )
    return {"nodes": nodes, "edges": edges}


def _enrich_with_s2(profile: ResearcherProfile, s2_profiles: list[ResearcherProfile], query: str) -> ResearcherProfile:
    ranked = rank_profiles(query, s2_profiles)
    if not ranked or ranked[0].match_score < 75:
        return profile
    best = ranked[0]
    profile.s2_id = profile.s2_id or best.s2_id
    profile.h_index = profile.h_index if profile.h_index is not None else best.h_index
    profile.affiliation = profile.affiliation or best.affiliation
    profile.cited_by_count = profile.cited_by_count if profile.cited_by_count is not None else best.cited_by_count
    profile.works_count = profile.works_count if profile.works_count is not None else best.works_count
    return profile


def collect_profile(name: str, *, openalex, semantic_scholar, pick: int | None, openalex_id: str | None, limit_works: int):
    errors: list[str] = []
    profile = None
    if openalex is not None and openalex_id:
        try:
            profile = openalex.get_author(openalex_id)
            profile.match_score = 100
        except SourceError as exc:
            errors.append(str(exc))
    if profile is None and openalex is not None:
        try:
            found = openalex.search_authors(name, limit=5)
        except SourceError as exc:
            errors.append(str(exc))
            found = []
        if found:
            profile = choose_profile(name, found, pick)
    if profile is None and semantic_scholar is not None:
        try:
            found = semantic_scholar.search_authors(name, limit=5)
            if found:
                profile = choose_profile(name, found, pick)
        except SourceError as exc:
            errors.append(str(exc))
    if profile is None:
        detail = "；".join(errors)
        raise NotFoundError(f"没有找到研究者：{name}" + (f"。{detail}" if detail else ""))
    if semantic_scholar is not None and not profile.s2_id:
        try:
            _enrich_with_s2(profile, semantic_scholar.search_authors(name, limit=3), name)
        except SourceError as exc:
            errors.append(str(exc))
    papers: list[Paper] = []
    if openalex is not None and profile.openalex_id:
        try:
            papers.extend(openalex.author_works(profile.openalex_id, limit=limit_works, sort="cited_by_count:desc"))
            papers.extend(openalex.author_works(profile.openalex_id, limit=limit_works, sort="publication_date:desc"))
        except SourceError as exc:
            errors.append(str(exc))
    if semantic_scholar is not None and profile.s2_id and len(papers) < 5:
        try:
            papers.extend(semantic_scholar.author_papers(profile.s2_id, limit=limit_works))
        except SourceError as exc:
            errors.append(str(exc))
    papers = deduplicate(papers, 92)
    related: list[ResearcherProfile] = []
    if openalex is not None and profile.topic_ids:
        try:
            related = openalex.related_authors(profile.topic_ids[0], limit=8)
        except SourceError as exc:
            errors.append(str(exc))
    return profile, papers, related, errors


def build_relations(profile: ResearcherProfile, papers: list[Paper], related: list[ResearcherProfile]) -> list[Relation]:
    relations = [
        Relation(other_name=name, relation="coauthor", weight=count, detail=f"采样论文中共同出现 {count} 次")
        for name, count in _coauthors(profile, papers)
    ]
    coauthor_names = {normalize_match(item.other_name) for item in relations}
    own = {normalize_match(profile.name), normalize_match(profile.openalex_id or "")}
    for person in related:
        if person.openalex_id and person.openalex_id == profile.openalex_id:
            continue
        if normalize_match(person.name) in own or normalize_match(person.name) in coauthor_names:
            continue
        detail = "、".join(person.concepts[:3]) or "主题相近"
        relations.append(
            Relation(
                other_name=person.name,
                relation="same_topic",
                weight=float(person.cited_by_count or 0),
                detail=detail,
                other_id=person.openalex_id,
            )
        )
    return relations


def profile_researcher(
    name: str,
    *,
    openalex,
    semantic_scholar,
    kb: KnowledgeBase,
    settings,
    llm=None,
    pick: int | None = None,
    openalex_id: str | None = None,
    limit_works: int = 30,
) -> Path:
    profile, papers, related, errors = collect_profile(
        name,
        openalex=openalex,
        semantic_scholar=semantic_scholar,
        pick=pick,
        openalex_id=openalex_id,
        limit_works=limit_works,
    )
    relations = build_relations(profile, papers, related)
    summary, directions = _fallback_summary(profile, papers)
    relations_note = "合作关系来自本次采样的论文，不是完整的职业生涯图谱。同方向研究者来自该研究者的首个 OpenAlex 主题。"
    if errors:
        relations_note += " 部分数据源没有返回：" + "；".join(errors)
    if llm is not None:
        paper_lines = "\n".join(
            f"- {paper.year or '未知'} {paper.title}（被引 {paper.citation_count or 0}）" for paper in papers[:20]
        )
        relation_lines = "\n".join(
            f"- {item.relation} {item.other_name} {int(item.weight)}" for item in relations[:20]
        )
        profile_text = (
            f"{profile.name}；机构 {profile.affiliation or '未知'}；"
            f"主题 {'、'.join(profile.concepts[:8]) or '无'}；"
            f"h-index {profile.h_index}；被引 {profile.cited_by_count}"
        )
        try:
            payload = llm.complete_json(
                RESEARCHER_SYSTEM,
                researcher_user_prompt(profile_text, paper_lines or "无", relation_lines or "无"),
                expect="object",
                max_tokens=1600,
            )
            if isinstance(payload, dict):
                summary = str(payload.get("summary") or summary)
                model_directions = payload.get("directions")
                if isinstance(model_directions, list) and model_directions:
                    directions = [str(item) for item in model_directions if str(item).strip()]
                if payload.get("relations_note"):
                    relations_note = str(payload["relations_note"])
        except LLMError as exc:
            relations_note += f" 模型画像生成失败：{exc}"
    researcher_id = researcher_storage_id(profile)
    report_dir = settings.data_dir / "reports" / "researchers"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{researcher_id}.md"
    graph_path = report_dir / f"{researcher_id}.graph.json"
    markdown = render_researcher(profile, papers, relations, summary, directions, relations_note)
    report_path.write_text(markdown, encoding="utf-8")
    graph_path.write_text(json.dumps(_graph(profile, relations), ensure_ascii=False, indent=2), encoding="utf-8")
    kb.upsert_researcher(researcher_id, profile, summary, directions, str(report_path))
    kb.replace_relations(researcher_id, relations)
    for paper in papers:
        try:
            key = kb.upsert_paper(paper)
        except ValueError:
            continue
        kb.link_researcher_paper(researcher_id, key)
    return report_path
