from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import click
import typer

from autopaper import __version__
from autopaper.analysis.biblio import fill_local_papers, format_recognized, recognize_uploaded
from autopaper.analysis.classify import reclassify_library
from autopaper.analysis.notes import (
    generate_author_profile,
    generate_topic_profile,
    note_papers,
    run_notes,
    write_paper_files,
)
from autopaper.analysis.pdfnotes import annotate_pdf
from autopaper.analysis.reader import build_reading, load_document, model_label
from autopaper.analysis.report import write_topic_report
from autopaper.analysis.researcher import profile_researcher
from autopaper.catalog import (
    format_author,
    require_existing_author,
    resolve_authors,
    resolve_topics,
    sorted_authors,
    sorted_topics,
    split_choices,
)
from autopaper.library import format_upload_report, plan_uploads, store_uploads
from autopaper.config import EXAMPLE_YAML, Settings, load_settings
from autopaper.errors import AmbiguousError, AutoPaperError, ConfigError, NotFoundError
from autopaper.http import HttpClient
from autopaper.llm.client import LLMClient
from autopaper.models import Paper, ScoredPaper, Topic
from autopaper.pipeline.collect import format_stats, run_collect
from autopaper.scheduler import due_topics
from autopaper.sources import available_source_names, build_sources, canonicalize_source
from autopaper.sources.semantic_scholar import s2_lookup_id
from autopaper.storage.db import KnowledgeBase
from autopaper.storage.export import export_topic, reference_list, write_bibtex
from autopaper.textutil import normalize_match, slugify

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="本地文献收集、筛选、知识库与科研阅读。",
)
topic_app = typer.Typer(no_args_is_help=True, add_completion=False, help="保存和修改研究主题。")
author_app = typer.Typer(no_args_is_help=True, add_completion=False, help="管理作者主名和别名。")
app.add_typer(topic_app, name="topic")
app.add_typer(author_app, name="author")

log = logging.getLogger(__name__)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def setup_logging(level: int, log_file: Path | None) -> None:
    logger = logging.getLogger("autopaper")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = logging.FileHandler(log_file, encoding="utf-8")
        handle.setFormatter(formatter)
        logger.addHandler(handle)


@contextmanager
def cli_errors():
    try:
        yield
    except typer.Exit:
        raise
    except (NotFoundError, AmbiguousError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    except AutoPaperError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None


def settings_from(ctx: typer.Context) -> Settings:
    settings = ctx.obj
    if not isinstance(settings, Settings):
        raise AutoPaperError("配置还没有加载")
    return settings


def option_given(ctx: typer.Context, name: str) -> bool:
    source = ctx.get_parameter_source(name)
    return source == click.core.ParameterSource.COMMANDLINE


def split_terms(values: list[str] | None) -> list[str]:
    terms: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                terms.append(part)
    return terms


def normalize_sources(values: list[str] | None) -> list[str]:
    sources = []
    for value in values or []:
        for part in str(value).split(","):
            name = canonicalize_source(part)
            if name and name not in sources:
                sources.append(name)
    return sources


def validate_topic(topic: Topic) -> None:
    if not topic.name.strip() or not topic.query.strip():
        raise AutoPaperError("主题名称和检索式不能为空")
    if topic.schedule not in {"manual", "daily", "weekly"}:
        raise AutoPaperError("schedule 必须是 manual、daily 或 weekly")
    if topic.sort not in {"relevance", "date"}:
        raise AutoPaperError("sort 必须是 relevance 或 date")
    if topic.keyword_mode not in {"any", "all"}:
        raise AutoPaperError("keyword-mode 必须是 any 或 all")
    if topic.domain_mode not in {"soft", "strict"}:
        raise AutoPaperError("domain-mode 必须是 soft 或 strict")
    known = set(available_source_names())
    unknown = [name for name in topic.sources if name not in known]
    if unknown:
        raise AutoPaperError(f"未知数据源：{', '.join(unknown)}。可选：{'、'.join(available_source_names())}")
    if not topic.sources:
        raise AutoPaperError("至少选择一个数据源")
    if topic.max_per_source < 1:
        raise AutoPaperError("每个数据源至少检索 1 篇")
    if topic.max_per_source > 200:
        topic.max_per_source = 200
    if topic.llm_top_n < 0:
        raise AutoPaperError("llm-top-n 不能为负数")
    if topic.min_score < 0 or topic.min_score > 100:
        raise AutoPaperError("min-score 需要在 0 到 100 之间")
    if topic.since_year is not None and not 1800 <= topic.since_year <= 2100:
        raise AutoPaperError("since-year 需要在 1800 到 2100 之间")


def maybe_llm(settings: Settings) -> LLMClient | None:
    if settings.disable_llm or not settings.llm_api_key:
        return None
    return LLMClient(settings)


def open_kb(settings: Settings) -> KnowledgeBase:
    return KnowledgeBase(settings.database_path)


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", help="输出调试日志。"),
    quiet: bool = typer.Option(False, "--quiet", help="只保留警告和错误日志。"),
) -> None:
    configure_stdio()
    try:
        settings = load_settings()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    level = logging.WARNING if quiet else logging.DEBUG if verbose else logging.INFO
    ctx.obj = settings
    if ctx.invoked_subcommand is None:
        return
    setup_logging(level, settings.data_dir / "logs" / "autopaper.log")


@app.command("init")
def init_cmd(ctx: typer.Context) -> None:
    """创建配置文件和本地知识库。"""
    with cli_errors():
        settings = settings_from(ctx)
        target = Path.cwd() / "autopaper.yaml"
        if target.exists():
            typer.echo(f"配置已存在：{target}")
        else:
            target.write_text(EXAMPLE_YAML, encoding="utf-8")
            typer.echo(f"已写入 {target}")
        env_example = Path.cwd() / ".env"
        if not env_example.exists():
            example = Path.cwd() / ".env.example"
            if example.exists():
                env_example.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
                typer.echo("已写入 .env，请填写 OPENAI_API_KEY。")
        with open_kb(settings) as kb:
            typer.echo(f"知识库：{kb.path}")
            typer.echo(f"当前有 {len(kb.list_topics())} 个主题。")


@app.command("doctor")
def doctor_cmd(
    ctx: typer.Context,
    online: bool = typer.Option(False, "--online", help="用一个很短的查询测试各个数据源。"),
) -> None:
    """检查配置、知识库和可选的网络连通性。"""
    with cli_errors():
        settings = settings_from(ctx)
        typer.echo(f"AutoPaper {__version__}")
        typer.echo(f"数据目录：{settings.data_dir}")
        typer.echo(f"知识库：{settings.database_path}")
        if not settings.mailto:
            typer.echo("建议在 autopaper.yaml 填写真实 mailto，以便走 OpenAlex / Crossref 礼貌池。")
        else:
            typer.echo(f"联系邮箱：{settings.mailto}")
        if settings.llm_api_key and not settings.disable_llm:
            typer.echo(f"模型：{settings.llm_model} @ {settings.llm_base_url}")
        else:
            typer.echo("未启用模型。检索和启发式排序仍可用，精读与摘要打分需要 OPENAI_API_KEY。")
        if settings.embedding_model and settings.embedding_api_key and not settings.disable_llm:
            typer.echo(f"语义召回：{settings.embedding_model} @ {settings.embedding_base_url}")
        else:
            typer.echo("未启用语义召回。字面主题词没命中的论文不会补进候选。")
        typer.echo("数据源：" + "、".join(available_source_names()))
        with open_kb(settings) as kb:
            typer.echo(f"主题 {len(kb.list_topics())} 个，库文件已就绪。")
        if not online:
            return
        with HttpClient.from_settings(settings) as http:
            sources = build_sources(available_source_names(), settings, http)
            for source in sources:
                try:
                    found = source.search("science", limit=1)
                    typer.echo(f"{source.name}：连通，示例 {len(found)} 篇")
                except Exception as exc:
                    typer.echo(f"{source.name}：失败，{exc}")


@topic_app.command("add")
def topic_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="主题名称，例如 graph neural networks for time series。"),
    query: Optional[str] = typer.Option(None, "--query", help="检索式。默认与名称相同。"),
    keyword: Optional[list[str]] = typer.Option(None, "--keyword", "-k", help="主题词，可重复，也可用逗号分隔。"),
    exclude: Optional[list[str]] = typer.Option(None, "--exclude", help="排除词，可重复。"),
    tag: Optional[list[str]] = typer.Option(None, "--tag", help="领域标签，可重复。"),
    domain_mode: str = typer.Option("soft", "--domain-mode", help="soft：没有分类的论文保留；strict：必须命中标签。"),
    source: Optional[list[str]] = typer.Option(None, "--source", help="数据源，可重复。默认五个都用。"),
    schedule: str = typer.Option("manual", "--schedule", help="manual、daily 或 weekly。"),
    sort: str = typer.Option("relevance", "--sort", help="relevance 或 date。"),
    max_per_source: int = typer.Option(20, "--max-per-source", help="每个数据源本次最多取多少篇，上限 200。"),
    since_year: Optional[int] = typer.Option(None, "--since-year", help="只看该年及以后的论文。"),
    min_score: float = typer.Option(0, "--min-score", help="低于该分数的论文会入库但标为 held。"),
    llm_top_n: int = typer.Option(20, "--llm-top-n", help="交给模型打分的篇数。0 表示不用模型。"),
    keyword_mode: str = typer.Option("any", "--keyword-mode", help="any：命中任一主题词；all：必须全部命中。"),
    topic_id: Optional[str] = typer.Option(None, "--id", help="自定义主题 id。"),
) -> None:
    """保存一个研究主题。"""
    with cli_errors():
        settings = settings_from(ctx)
        if max_per_source > 200:
            typer.echo("每个数据源单次最多检索 200 篇，已按 200 处理。")
            max_per_source = 200
        chosen_id = topic_id or slugify(name)
        topic = Topic(
            id=chosen_id,
            name=name.strip(),
            query=(query or name).strip(),
            keywords=split_terms(keyword),
            exclude_keywords=split_terms(exclude),
            domain_tags=split_terms(tag),
            domain_mode=domain_mode,
            sources=normalize_sources(source) or available_source_names(),
            schedule=schedule,
            sort=sort,
            max_per_source=max_per_source,
            since_year=since_year,
            min_score=min_score,
            llm_top_n=llm_top_n,
            keyword_mode=keyword_mode,
        )
        validate_topic(topic)
        with open_kb(settings) as kb:
            for existing in kb.list_topics():
                if existing.id == topic.id or normalize_match(existing.name) == normalize_match(topic.name):
                    raise AutoPaperError(f"主题已存在：{existing.id}")
            kb.save_topic(topic)
        typer.echo(f"已保存主题 {topic.id}")
        typer.echo(f"检索式：{topic.query}")
        typer.echo("下一步：autopaper collect " + topic.id)


@topic_app.command("list")
def topic_list(ctx: typer.Context) -> None:
    """列出已保存的主题。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            topics = kb.list_topics()
        if not topics:
            typer.echo("还没有主题。用 autopaper topic add 创建一个。")
            return
        for topic in topics:
            last = topic.last_run_at.isoformat() if topic.last_run_at else "尚未运行"
            typer.echo(f"{topic.id}\t{topic.schedule}\t{last}\t{topic.name}")


@topic_app.command("show")
def topic_show(ctx: typer.Context, topic_ref: str = typer.Argument(..., help="主题 id 或名称。")) -> None:
    """查看一个主题的检索设置。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            topic = kb.get_topic(topic_ref)
        typer.echo(f"id: {topic.id}")
        typer.echo(f"name: {topic.name}")
        typer.echo(f"query: {topic.query}")
        typer.echo(f"keywords: {', '.join(topic.keywords) or '-'}")
        typer.echo(f"exclude: {', '.join(topic.exclude_keywords) or '-'}")
        typer.echo(f"tags: {', '.join(topic.domain_tags) or '-'}")
        typer.echo(f"domain_mode: {topic.domain_mode}")
        typer.echo(f"sources: {', '.join(topic.sources)}")
        typer.echo(f"schedule: {topic.schedule}")
        typer.echo(f"sort: {topic.sort}")
        typer.echo(f"max_per_source: {topic.max_per_source}")
        typer.echo(f"since_year: {topic.since_year or '-'}")
        typer.echo(f"min_score: {topic.min_score}")
        typer.echo(f"llm_top_n: {topic.llm_top_n}")
        typer.echo(f"keyword_mode: {topic.keyword_mode}")
        typer.echo(f"last_run_at: {topic.last_run_at.isoformat() if topic.last_run_at else '-'}")


@topic_app.command("edit")
def topic_edit(
    ctx: typer.Context,
    topic_ref: str = typer.Argument(..., help="主题 id 或名称。"),
    query: Optional[str] = typer.Option(None, "--query"),
    keyword: Optional[list[str]] = typer.Option(None, "--keyword", "-k"),
    exclude: Optional[list[str]] = typer.Option(None, "--exclude"),
    tag: Optional[list[str]] = typer.Option(None, "--tag"),
    domain_mode: Optional[str] = typer.Option(None, "--domain-mode"),
    source: Optional[list[str]] = typer.Option(None, "--source"),
    schedule: Optional[str] = typer.Option(None, "--schedule"),
    sort: Optional[str] = typer.Option(None, "--sort"),
    max_per_source: Optional[int] = typer.Option(None, "--max-per-source"),
    since_year: Optional[int] = typer.Option(None, "--since-year"),
    min_score: Optional[float] = typer.Option(None, "--min-score"),
    llm_top_n: Optional[int] = typer.Option(None, "--llm-top-n"),
    keyword_mode: Optional[str] = typer.Option(None, "--keyword-mode"),
    clear_keywords: bool = typer.Option(False, "--clear-keywords"),
    clear_exclude: bool = typer.Option(False, "--clear-exclude"),
    clear_tags: bool = typer.Option(False, "--clear-tags"),
) -> None:
    """修改主题。只更新你传入的字段。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            topic = kb.get_topic(topic_ref)
            if query is not None:
                topic.query = query.strip()
            if option_given(ctx, "keyword"):
                topic.keywords = split_terms(keyword)
            if option_given(ctx, "exclude"):
                topic.exclude_keywords = split_terms(exclude)
            if option_given(ctx, "tag"):
                topic.domain_tags = split_terms(tag)
            if clear_keywords:
                topic.keywords = []
            if clear_exclude:
                topic.exclude_keywords = []
            if clear_tags:
                topic.domain_tags = []
            if domain_mode is not None:
                topic.domain_mode = domain_mode
            if option_given(ctx, "source"):
                topic.sources = normalize_sources(source)
            if schedule is not None:
                topic.schedule = schedule
            if sort is not None:
                topic.sort = sort
            if max_per_source is not None:
                if max_per_source > 200:
                    typer.echo("每个数据源单次最多检索 200 篇，已按 200 处理。")
                    max_per_source = 200
                topic.max_per_source = max_per_source
            if since_year is not None:
                topic.since_year = since_year
            if min_score is not None:
                topic.min_score = min_score
            if llm_top_n is not None:
                topic.llm_top_n = llm_top_n
            if keyword_mode is not None:
                topic.keyword_mode = keyword_mode
            validate_topic(topic)
            kb.save_topic(topic)
        typer.echo(f"已更新主题 {topic.id}")


@topic_app.command("rm")
def topic_rm(
    ctx: typer.Context,
    topic_ref: str = typer.Argument(..., help="主题 id 或名称。"),
    yes: bool = typer.Option(False, "--yes", "-y", help="不再询问确认。"),
) -> None:
    """删除主题。论文仍留在知识库里。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            topic = kb.get_topic(topic_ref)
            if not yes and not typer.confirm(f"删除主题 {topic.name}？论文仍会留在知识库中。"):
                raise typer.Exit()
            kb.delete_topic(topic.id)
        typer.echo(f"已删除主题 {topic.id}")


@app.command("collect")
def collect_cmd(
    ctx: typer.Context,
    topic_ref: str = typer.Argument(..., help="主题 id 或名称。"),
    fresh: bool = typer.Option(False, "--fresh", help="先清空该主题已有关联，再写入本次结果。"),
    with_report: bool = typer.Option(False, "--with-report", help="收集结束后写研究报告。"),
) -> None:
    """按主题检索、去重、过滤、打分并写入知识库。"""
    with cli_errors():
        settings = settings_from(ctx)
        llm = maybe_llm(settings)
        if llm is None:
            typer.echo("未配置 OPENAI_API_KEY，或已关闭模型。本次使用启发式相关性打分。")
        try:
            with open_kb(settings) as kb:
                topic = kb.get_topic(topic_ref)
                with HttpClient.from_settings(settings) as http:
                    sources = build_sources(topic.sources, settings, http)
                    stats = run_collect(
                        topic,
                        kb,
                        settings,
                        sources=sources,
                        llm=llm,
                        fresh=fresh,
                        export=True,
                        with_report=with_report,
                    )
            typer.echo(format_stats(stats))
            if stats.status == "failed":
                raise typer.Exit(code=1)
        finally:
            if llm is not None:
                llm.close()


@app.command("papers")
def papers_cmd(
    ctx: typer.Context,
    topic_ref: str = typer.Argument(..., help="主题 id 或名称。"),
    limit: int = typer.Option(30, "--limit", help="最多显示多少篇。"),
    show_all: bool = typer.Option(False, "--all", help="包括低于最低分的论文。"),
) -> None:
    """查看主题下已入库的论文。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            topic = kb.get_topic(topic_ref)
            items = kb.list_topic_papers(topic.id, include_held=show_all)
        if not items:
            typer.echo("这个主题下还没有论文。")
            return
        for item in items[:limit]:
            paper = item.paper
            typer.echo(f"{item.score:6.1f}  {paper.year or '-':4}  {paper.citation_count or 0:5}  {paper.title}")
            typer.echo(f"        {paper.key}")
        if len(items) > limit:
            typer.echo(f"还有 {len(items) - limit} 篇，用 --limit 提高数量。")


@app.command("export")
def export_cmd(
    ctx: typer.Context,
    topic_ref: str = typer.Argument(..., help="主题 id 或名称。"),
    fmt: str = typer.Option("all", "--format", "-f", help="csv、json、bib、md 或 all。"),
    show_all: bool = typer.Option(False, "--all", help="包括低于最低分的论文。"),
) -> None:
    """把主题文献导出为 CSV、JSON、BibTeX 或 Markdown。"""
    with cli_errors():
        settings = settings_from(ctx)
        formats = {"csv", "json", "bib", "md"} if fmt == "all" else {part.strip() for part in fmt.split(",") if part.strip()}
        allowed = {"csv", "json", "bib", "bibtex", "md", "markdown"}
        unknown = formats - allowed
        if unknown:
            raise AutoPaperError(f"不认识的导出格式：{', '.join(sorted(unknown))}")
        with open_kb(settings) as kb:
            topic = kb.get_topic(topic_ref)
            items = kb.list_topic_papers(topic.id, include_held=show_all)
            dest = settings.data_dir / "exports" / topic.id / "manual"
            written = export_topic(dest, topic.name, items, formats)
        if not written:
            raise AutoPaperError("没有写出文件")
        for path in written.values():
            typer.echo(path)


def _prompt_authors(kb: KnowledgeBase) -> str:
    authors = sorted_authors(kb)
    if authors:
        typer.echo("已有作者：")
        for index, author in enumerate(authors, start=1):
            typer.echo(f"  {index}. {format_author(author)}")
    else:
        typer.echo("还没有作者。输入主名即可新建。")
    return typer.prompt("作者（多项用 ; 隔开，可填编号、主名、别名或新名字）")


def _prompt_topics(kb: KnowledgeBase) -> str:
    topics = sorted_topics(kb)
    if topics:
        typer.echo("已有主题：")
        for index, topic in enumerate(topics, start=1):
            typer.echo(f"  {index}. {topic.name}")
    else:
        typer.echo("还没有主题。输入名称即可新建。")
    return typer.prompt("研究主题（多项用 ; 隔开，可填编号、已有名称或新名称）")


@app.command("dedupe")
def dedupe_cmd(ctx: typer.Context) -> None:
    """按题名、副标题、DOI 和其他标识合并库里的重复论文。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            groups = kb.duplicate_groups()
            if not groups:
                typer.echo("没有发现重复论文。")
                return
            for group in groups:
                titles = "、".join(paper.title for paper in group)
                typer.echo(f"将合并：{titles}")
            merged = kb.merge_same_papers()
        for item in merged:
            typer.echo(f"已并入《{item['title']}》")


@app.command("upload")
def upload_cmd(
    ctx: typer.Context,
    directory: Path = typer.Argument(..., help="存放论文 PDF、TXT 或 Markdown 的目录。"),
    author: Optional[str] = typer.Option(None, "--author", "-a", help="作者。多项用 ; 隔开，可填编号、主名、别名或新名字。"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="研究主题。多项用 ; 隔开，可填编号、已有名称或新名称。"),
) -> None:
    """把目录中的论文按题名、副标题、DOI 和其他标识归类入库。同一篇会跳过。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            planned = plan_uploads(directory, kb, settings.data_dir)
            incoming = [item for item in planned if not item.skip]
            authors = []
            topics = []
            uploaded = []
            if incoming:
                author_raw = author if author is not None else _prompt_authors(kb)
                topic_raw = topic if topic is not None else _prompt_topics(kb)
                if not split_choices(author_raw):
                    raise AutoPaperError("请至少指定一位作者")
                if not split_choices(topic_raw):
                    raise AutoPaperError("请至少指定一个研究主题")
                authors = resolve_authors(kb, author_raw)
                topics = resolve_topics(kb, topic_raw)
                planned = store_uploads(planned, kb, authors, topics)
                uploaded = [kb.resolve_paper(item.paper_key) for item in planned if not item.skip and item.paper_key]
                uploaded = [paper for paper in uploaded if paper is not None]
                uploaded, records = recognize_uploaded(kb, uploaded)
            typer.echo(format_upload_report(planned, authors, topics))
            for record in records if incoming else []:
                typer.echo(f"{record['title']}  {format_recognized(record)}")
            if uploaded:
                llm = maybe_llm(settings)
                if llm is None:
                    typer.echo("未配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY，已归类但没有生成笔记。")
                else:
                    try:
                        note_papers(kb, settings.data_dir, llm, uploaded, echo=typer.echo)
                    finally:
                        llm.close()


@app.command("biblio")
def biblio_cmd(ctx: typer.Context) -> None:
    """从已归档论文原文识别年份、期刊或会议、DOI 和链接。只补空白字段，不调用模型。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            records = fill_local_papers(kb)
        filled = sum(1 for record in records if record["filled"])
        for record in records:
            typer.echo(f"{record['title']}  {format_recognized(record)}")
        typer.echo(f"已检查 {len(records)} 篇，写入 {filled} 篇原先空着的题录。")


@author_app.command("list")
def author_list(ctx: typer.Context) -> None:
    """列出作者。别名显示在主名后面的括号中。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            authors = sorted_authors(kb)
        if not authors:
            typer.echo("还没有作者。上传论文或使用 autopaper author add 可以新建。")
            return
        for author in authors:
            typer.echo(f"{author.id}\t{format_author(author)}")


@author_app.command("add")
def author_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="主名，例如 Huilian Sophie Qiu。"),
    alias: Optional[list[str]] = typer.Option(None, "--alias", help="别名，可重复。例如 Huilian Qiu 或 邱惠莲。"),
) -> None:
    """新建作者，并可以同时写入别名。"""
    with cli_errors():
        settings = settings_from(ctx)
        aliases = []
        for value in alias or []:
            aliases.extend(split_choices(value))
        with open_kb(settings) as kb:
            author = kb.create_author(name, aliases)
        typer.echo(f"已保存作者 {format_author(author)}")


@author_app.command("alias")
def author_alias(
    ctx: typer.Context,
    author_ref: str = typer.Argument(..., help="作者 id、主名或已有别名。"),
    add: Optional[list[str]] = typer.Option(None, "--add", help="增加别名，可重复，也可用 ; 隔开。"),
    remove: Optional[list[str]] = typer.Option(None, "--remove", help="删除别名，可重复，也可用 ; 隔开。"),
) -> None:
    """手动调整作者别名。"""
    with cli_errors():
        settings = settings_from(ctx)
        added = []
        removed = []
        for value in add or []:
            added.extend(split_choices(value))
        for value in remove or []:
            removed.extend(split_choices(value))
        with open_kb(settings) as kb:
            author = require_existing_author(kb, author_ref)
            if added or removed:
                author = kb.update_aliases(author.id, add=added, remove=removed)
        typer.echo(format_author(author))


@app.command("search")
def search_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="在本地知识库的标题、摘要、作者和 DOI 中查找。"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """搜索已经入库的论文。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            papers = kb.search_papers(query, limit=limit)
        if not papers:
            typer.echo("知识库里没有匹配的论文。")
            return
        for paper in papers:
            typer.echo(f"{paper.year or '-'}  {paper.title}")
            typer.echo(f"    {paper.key}")


@app.command("classify")
def classify_cmd(ctx: typer.Context) -> None:
    """按已有主题为本地论文再分类。有模型时用模型，否则按主题词匹配。"""
    with cli_errors():
        settings = settings_from(ctx)
        llm = maybe_llm(settings)
        try:
            with open_kb(settings) as kb:
                result = reclassify_library(kb, llm)
            typer.echo(result["message"])
            for item in result["links"]:
                typer.echo(f"  {item['paper']} -> {item['topic']}")
        finally:
            if llm is not None:
                llm.close()


@app.command("read")
def read_cmd(
    ctx: typer.Context,
    paper_ref: Optional[str] = typer.Argument(None, help="知识库中的论文 key、DOI、arXiv id 或标题。"),
    file: Optional[Path] = typer.Option(None, "--file", help="本地 PDF 或文本文件。"),
    title: Optional[str] = typer.Option(None, "--title", help="文件还没入库时使用的标题。"),
    topic_ref: Optional[str] = typer.Option(None, "--topic", help="把这篇论文关联到一个主题。"),
) -> None:
    """按固定科研阅读问题精读一篇论文，并写入知识库。"""
    with cli_errors():
        settings = settings_from(ctx)
        llm = maybe_llm(settings)
        if llm is None:
            raise AutoPaperError("精读需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。检索和研究者题录整理不依赖它。")
        try:
            with open_kb(settings) as kb:
                paper = _resolve_or_create_paper(kb, paper_ref, file, title)
                document, only_abstract, source = _reading_text(paper, file)
                payload, markdown = build_reading(
                    paper,
                    document,
                    llm,
                    only_abstract=only_abstract,
                    source_name=source.name if source else "",
                )
                kb.save_reading(paper.key, payload, markdown)
                library_note = write_paper_files(paper, payload, markdown, settings.llm_model)
                if topic_ref:
                    topic = kb.get_topic(topic_ref)
                    kb.link_topic_paper(
                        topic.id,
                        ScoredPaper(paper, 100, None, 100, "手动阅读", "candidate"),
                        paper.key,
                    )
                path = _write_reading_files(settings, paper, payload, markdown)
            typer.echo(f"已写入 {path}", err=True)
            if library_note is not None:
                typer.echo(f"已写入 {library_note}", err=True)
            typer.echo(markdown)
        finally:
            llm.close()


@app.command("annotate")
def annotate_cmd(
    ctx: typer.Context,
    paper_ref: str = typer.Argument(..., help="知识库中的论文 key、DOI、arXiv id 或标题。"),
) -> None:
    """在论文 PDF 原件上写入彩色批注。"""
    with cli_errors():
        settings = settings_from(ctx)
        llm = maybe_llm(settings)
        if llm is None:
            raise AutoPaperError("自动批注需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。")
        try:
            with open_kb(settings) as kb:
                paper = _resolve_or_create_paper(kb, paper_ref, None, None)
                if not paper.local_path:
                    raise AutoPaperError("这篇论文没有本地 PDF。")
                path = Path(paper.local_path)
                if not path.exists():
                    raise NotFoundError("本地文件已经不在原位置")
                if path.suffix.lower() != ".pdf":
                    raise AutoPaperError("自动批注目前只写入 PDF。")
                result = annotate_pdf(path, llm)
            typer.echo(result.message())
        finally:
            llm.close()


@app.command("profile")
def profile_cmd(
    ctx: typer.Context,
    author: Optional[str] = typer.Option(None, "--author", "-a", help="为这位作者写研究方向、定位和学术关系。"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="为这个主题写概况，并写出其中每位作者的方向、定位和学术关系。"),
) -> None:
    """根据已有单篇笔记，生成作者笔记或主题笔记。"""
    with cli_errors():
        if not author and not topic:
            raise AutoPaperError("请指定 --author 或 --topic。")
        settings = settings_from(ctx)
        llm = maybe_llm(settings)
        if llm is None:
            raise AutoPaperError("生成笔记需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。")
        try:
            with open_kb(settings) as kb:
                typer.echo(f"模型：{model_label(settings.llm_model)}")
                if author:
                    person = require_existing_author(kb, author)
                    path = generate_author_profile(kb, settings.data_dir, llm, person)
                    typer.echo(path)
                if topic:
                    chosen = kb.get_topic(topic)
                    path = generate_topic_profile(kb, settings.data_dir, llm, chosen)
                    typer.echo(path)
        finally:
            llm.close()


@app.command("notes")
def notes_cmd(
    ctx: typer.Context,
    author: Optional[str] = typer.Option(None, "--author", "-a", help="只阅读这位作者已归类的论文。可填主名、别名或编号。"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="只阅读这个主题下已归类的论文。"),
) -> None:
    """用模型阅读已归类论文，在论文目录写 note.md，并按作者和主题写综合笔记。"""
    with cli_errors():
        settings = settings_from(ctx)
        llm = maybe_llm(settings)
        if llm is None:
            raise AutoPaperError("生成笔记需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。")
        try:
            with open_kb(settings) as kb:
                chosen_author = require_existing_author(kb, author) if author else None
                chosen_topic = kb.get_topic(topic) if topic else None
                typer.echo(f"模型：{model_label(settings.llm_model)}")
                written = run_notes(
                    kb,
                    settings.data_dir,
                    llm,
                    author=chosen_author,
                    topic=chosen_topic,
                    echo=typer.echo,
                )
            for path in written:
                typer.echo(path)
        finally:
            llm.close()


def _resolve_or_create_paper(kb: KnowledgeBase, paper_ref: str | None, file: Path | None, title: str | None) -> Paper:
    paper = None
    if paper_ref:
        paper = kb.resolve_paper(paper_ref)
        if paper is None:
            hits = kb.search_papers(paper_ref, limit=5)
            if len(hits) == 1:
                paper = hits[0]
            elif len(hits) > 1:
                lines = ["找到多篇论文，请用 key 指定："]
                lines.extend(f"- {item.key}  {item.title}" for item in hits)
                raise AmbiguousError("\n".join(lines))
            elif file is None:
                raise NotFoundError(f"知识库里没有这篇论文：{paper_ref}")
    if paper is None:
        if file is None and not title:
            raise AutoPaperError("请提供论文 key、DOI，或用 --file 指向 PDF / 文本。")
        paper_title = title or (file.stem if file else paper_ref or "未命名论文")
        created = Paper(title=paper_title, source="local", sources=["local"])
        key = kb.upsert_paper(created)
        paper = kb.resolve_paper(key)
        if paper is None:
            raise AutoPaperError("论文没有写入知识库")
    return paper


def _reading_text(paper: Paper, file: Path | None) -> tuple[str, bool, Path | None]:
    path = file
    if path is None and paper.local_path:
        candidate = Path(paper.local_path)
        if candidate.exists():
            path = candidate
    if path is not None:
        text = load_document(path).strip()
        if not text:
            raise AutoPaperError("文件里没有提取到文本")
        return text, False, path
    if paper.abstract.strip():
        return paper.abstract, True, None
    raise AutoPaperError("这篇论文只有题录，没有摘要。请用 --file 提供 PDF 或文本。")


def _write_reading_files(settings: Settings, paper: Paper, payload: dict, markdown: str) -> Path:
    import json

    safe = paper.key.replace(":", "_")
    directory = settings.data_dir / "reports" / "readings"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe}.md"
    path.write_text(markdown, encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@app.command("report")
def report_cmd(ctx: typer.Context, topic_ref: str = typer.Argument(..., help="主题 id 或名称。")) -> None:
    """根据知识库中的论文生成研究报告和引用列表。"""
    with cli_errors():
        settings = settings_from(ctx)
        llm = maybe_llm(settings)
        try:
            with open_kb(settings) as kb:
                topic = kb.get_topic(topic_ref)
                path = write_topic_report(kb, topic, settings, llm)
            typer.echo(f"已写入 {path}", err=True)
            typer.echo(path.read_text(encoding="utf-8"))
        finally:
            if llm is not None:
                llm.close()


@app.command("researcher")
def researcher_cmd(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="研究者姓名。加 --list 时可以省略。"),
    list_profiles: bool = typer.Option(False, "--list", help="列出已经生成过的研究者画像。"),
    pick: Optional[int] = typer.Option(None, "--pick", help="同名研究者的序号，从 1 开始。"),
    openalex_id: Optional[str] = typer.Option(None, "--openalex-id", help="直接指定 OpenAlex 作者编号，例如 A5023888391。"),
    limit_works: int = typer.Option(30, "--limit-works", help="高被引和近年各采样多少篇。"),
) -> None:
    """整理一位研究者的方向、代表论文，以及合作者和同方向研究者。"""
    with cli_errors():
        settings = settings_from(ctx)
        if list_profiles or not name:
            with open_kb(settings) as kb:
                rows = kb.list_researchers()
            if not rows:
                typer.echo("还没有研究者画像。")
                return
            for row in rows:
                typer.echo(f"{row['id']}\t{row['name']}\t{row['affiliation'] or ''}")
            return
        llm = maybe_llm(settings)
        try:
            with open_kb(settings) as kb, HttpClient.from_settings(settings) as http:
                sources = {source.name: source for source in build_sources(["openalex", "semantic_scholar"], settings, http)}
                path = profile_researcher(
                    name,
                    openalex=sources["openalex"],
                    semantic_scholar=sources["semantic_scholar"],
                    kb=kb,
                    settings=settings,
                    llm=llm,
                    pick=pick,
                    openalex_id=openalex_id,
                    limit_works=limit_works,
                )
            typer.echo(f"已写入 {path}", err=True)
            typer.echo(path.read_text(encoding="utf-8"))
        finally:
            if llm is not None:
                llm.close()


@app.command("citations")
def citations_cmd(
    ctx: typer.Context,
    paper_ref: str = typer.Argument(..., help="论文 key、DOI 或标题。"),
    limit: int = typer.Option(30, "--limit"),
) -> None:
    """从 Semantic Scholar 拉取一篇论文的参考文献，并写成 BibTeX。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            paper = kb.resolve_paper(paper_ref)
            if paper is None:
                hits = kb.search_papers(paper_ref, limit=5)
                if len(hits) == 1:
                    paper = hits[0]
                elif len(hits) > 1:
                    raise AmbiguousError("\n".join(["找到多篇论文，请用 key 指定：", *[item.key for item in hits]]))
                else:
                    raise NotFoundError(f"知识库里没有这篇论文：{paper_ref}")
            lookup = s2_lookup_id(paper)
            if not lookup:
                raise AutoPaperError("这篇论文没有 DOI、arXiv id 或 Semantic Scholar id，无法拉取参考文献。")
            with HttpClient.from_settings(settings) as http:
                source = build_sources(["semantic_scholar"], settings, http)[0]
                references = source.references(lookup, limit=limit)
            safe = paper.key.replace(":", "_")
            dest = settings.data_dir / "exports" / "citations"
            dest.mkdir(parents=True, exist_ok=True)
            bib_path = dest / f"{safe}.bib"
            md_path = dest / f"{safe}.md"
            write_bibtex(bib_path, references)
            md_path.write_text(
                f"# {paper.title} 的参考文献\n\n" + reference_list(references) + "\n",
                encoding="utf-8",
            )
        typer.echo(f"已写入 {bib_path}")
        typer.echo(f"已写入 {md_path}")
        typer.echo(reference_list(references))


schedule_app = typer.Typer(no_args_is_help=True, add_completion=False, help="按保存的周期运行主题。")
app.add_typer(schedule_app, name="schedule")


@schedule_app.command("run")
def schedule_run(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run", help="只列出到期主题，不访问数据源。"),
    include_all: bool = typer.Option(False, "--all", help="忽略周期，运行全部主题。"),
) -> None:
    """运行已到期的每日或每周主题。加上 --all 时连手动主题一起运行。"""
    with cli_errors():
        settings = settings_from(ctx)
        with open_kb(settings) as kb:
            selected = due_topics(kb.list_topics(), include_all=include_all)
            if not selected:
                typer.echo("没有到期的主题。")
                return
            if dry_run:
                for topic in selected:
                    typer.echo(f"到期：{topic.id}（{topic.schedule}）")
                return
            llm = maybe_llm(settings)
            failed = False
            try:
                with HttpClient.from_settings(settings) as http:
                    for topic in selected:
                        typer.echo(f"开始 {topic.id}")
                        sources = build_sources(topic.sources, settings, http)
                        stats = run_collect(topic, kb, settings, sources=sources, llm=llm, export=True)
                        typer.echo(format_stats(stats))
                        if stats.status == "failed":
                            failed = True
            finally:
                if llm is not None:
                    llm.close()
        if failed:
            raise typer.Exit(code=1)


@app.command("serve")
def serve_cmd(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", help="只监听本机。"),
    port: int = typer.Option(8765, "--port", help="端口。"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="启动后是否打开浏览器。"),
) -> None:
    """打开本地图谱界面，用来查看相关度、阅读笔记和上传论文。"""
    with cli_errors():
        settings = settings_from(ctx)
        from autopaper.web.server import serve

        serve(settings, host, port, open_browser)


@app.command("version")
def version_cmd() -> None:
    """显示版本。"""
    typer.echo(__version__)


def main() -> None:
    app()
