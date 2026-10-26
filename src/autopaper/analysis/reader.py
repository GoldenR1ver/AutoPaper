from __future__ import annotations

from pathlib import Path

from autopaper.errors import AutoPaperError, NotFoundError
from autopaper.llm.prompts import READING_SYSTEM, reading_user_prompt

MODEL_LABELS = {
    "deepseek-flash": "DeepSeek-V4.1-Flash（deepseek-flash）",
}
from autopaper.models import Paper


def load_document(path: Path) -> str:
    if not path.exists():
        raise NotFoundError(f"找不到文件：{path}")
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    return _read_text(path)


def load_catalog_source(path: Path) -> tuple[str, str]:
    """返回正文，以及 PDF 元数据里的作者。归类时用作者栏，不把全文交给模型。"""
    if not path.exists():
        raise NotFoundError(f"找不到文件：{path}")
    if path.suffix.lower() != ".pdf":
        return _read_text(path), ""
    reader = _open_pdf(path)
    author = ""
    meta = reader.metadata
    if meta is not None and meta.author:
        author = str(meta.author)
    return _pdf_pages(reader), author


def extract_pdf(path: Path) -> str:
    return _pdf_pages(_open_pdf(path))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AutoPaperError(f"无法读取文件：{exc}") from exc


def _open_pdf(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AutoPaperError("读取 PDF 需要安装 pypdf") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise AutoPaperError("这份 PDF 已加密，无法提取文本")
    except AutoPaperError:
        raise
    except Exception as exc:
        raise AutoPaperError(f"无法读取 PDF：{exc}") from exc
    return reader


def _pdf_pages(reader) -> str:
    try:
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise AutoPaperError(f"无法读取 PDF：{exc}") from exc
    return "\n\n".join(pages)


def clip_paper_text(text: str, limit: int = 120000) -> str:
    body = text.strip()
    if len(body) <= limit:
        return body
    head = int(limit * 0.8)
    tail = limit - head
    return body[:head] + "\n\n[...中间内容已省略...]\n\n" + body[-tail:]


def paper_meta(paper: Paper) -> str:
    return "\n".join(
        [
            f"标题：{paper.title}",
            f"作者：{', '.join(paper.authors) or '未知'}",
            f"年份：{paper.year or '未知'}",
            f"期刊或会议：{paper.venue or '未知'}",
            f"DOI：{paper.doi or '无'}",
            f"链接：{paper.url or paper.pdf_url or '无'}",
        ]
    )


def _lines(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _bullets(value) -> str:
    lines = _lines(value)
    if not lines:
        return "论文文本未说明"
    if len(lines) == 1 and not isinstance(value, list):
        return lines[0]
    return "\n".join(f"- {line}" for line in lines)


def _axis_block(value) -> str:
    if not isinstance(value, dict):
        return _bullets(value)
    labels = (("method", "方法"), ("direction", "方向"), ("theory", "理论"))
    used = {key for key, _label in labels}
    chunks = []
    for key, label in labels:
        body = _lines(value.get(key))
        if body:
            chunks.append(f"### {label}\n\n" + "\n".join(f"- {line}" for line in body))
    for key, item in value.items():
        if key in used:
            continue
        body = _lines(item)
        if body:
            chunks.append(f"### {key}\n\n" + "\n".join(f"- {line}" for line in body))
    return "\n\n".join(chunks) if chunks else "论文文本未说明"


def _reproduction_block(value) -> str:
    if not isinstance(value, dict):
        return _bullets(value)
    steps = _lines(value.get("steps"))
    step_text = "\n".join(f"- {line}" for line in steps) if steps else "论文文本未说明"
    data = str(value.get("data") or "").strip() or "论文文本未说明"
    risks = str(value.get("risks") or "").strip() or "论文文本未说明"
    return f"数据与材料：{data}\n\n步骤：\n{step_text}\n\n风险：{risks}"


def _claims_block(value) -> str:
    if not isinstance(value, list):
        return _bullets(value)
    rows = []
    for item in value:
        if isinstance(item, dict):
            claim = str(item.get("claim") or "").strip()
            evidence = str(item.get("evidence") or "").strip() or "论文文本未说明"
            where = str(item.get("where") or "").strip() or "论文文本未说明"
            if claim:
                rows.append(f"- 主张：{claim}。证据：{evidence}。位置：{where}")
        else:
            text = str(item).strip()
            if text:
                rows.append(f"- {text}")
    return "\n".join(rows) if rows else "论文文本未说明"


def _terms_block(value) -> str:
    if not isinstance(value, list):
        return _bullets(value)
    rows = []
    for item in value:
        if isinstance(item, dict):
            term = str(item.get("term") or "").strip()
            meaning = str(item.get("meaning") or "").strip() or "论文文本未说明"
            if term:
                rows.append(f"- {term}：{meaning}")
        else:
            text = str(item).strip()
            if text:
                rows.append(f"- {text}")
    return "\n".join(rows) if rows else "论文文本未说明"


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


def model_label(model: str) -> str:
    if not model:
        return ""
    return MODEL_LABELS.get(model, model)


def render_reading(paper: Paper, payload: dict, *, model: str = "", source_name: str = "") -> str:
    title = str(payload.get("title") or "").strip() or paper.title
    header = [f"# {title}", ""]
    label = model_label(model)
    if label:
        header.append(f"模型：{label}")
    if source_name:
        header.append(f"原文：{source_name}")
    header.extend(["", paper_meta(paper), ""])
    sections = [
        _section("一句话", _bullets(payload.get("one_sentence"))),
        _section("所在领域", _bullets(payload.get("field"))),
        _section("领域发展状况", _bullets(payload.get("field_status"))),
        _section("知识网络中的坐标", _bullets(payload.get("position"))),
        _section("背景", _bullets(payload.get("background"))),
        _section("提出的问题", _bullets(payload.get("problem"))),
        _section("方法", _bullets(payload.get("method"))),
        _section("实验", _bullets(payload.get("experiments"))),
        _section("结果", _bullets(payload.get("results"))),
        _section("能否解决该问题", _bullets(payload.get("solves_problem"))),
        _section("新的问题", _bullets(payload.get("new_questions"))),
        _section("创新点", _axis_block(payload.get("innovations"))),
        _section("可能的不足", _axis_block(payload.get("limitations"))),
        _section("复现思路", _reproduction_block(payload.get("reproduction"))),
        _section("主张与证据", _claims_block(payload.get("claims"))),
        _section("关键术语", _terms_block(payload.get("terms"))),
        _section("需要核对的地方", _bullets(payload.get("reader_checks"))),
    ]
    return "\n".join(header + sections)


def build_reading(paper: Paper, text: str, llm, *, only_abstract: bool, source_name: str = "") -> tuple[dict, str]:
    clipped = clip_paper_text(text)
    payload = llm.complete_json(
        READING_SYSTEM,
        reading_user_prompt(paper_meta(paper), clipped, only_abstract),
        expect="object",
        max_tokens=7000,
    )
    if not isinstance(payload, dict):
        payload = {}
    model = getattr(getattr(llm, "settings", None), "llm_model", "")
    markdown = render_reading(paper, payload, model=model, source_name=source_name)
    return payload, markdown
