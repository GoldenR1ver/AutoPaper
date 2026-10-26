from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from autopaper.errors import AutoPaperError
from autopaper.llm.prompts import ANNOTATE_SYSTEM, annotate_metric_prompt, annotate_user_prompt
from autopaper.analysis.reader import clip_paper_text

KIND_COLOR = {
    "keyword": "fdf485",
    "keyword_note": "fff3a0",
    "definition": "ffd200",
    "related": "38e5ff",
    "method": "ff809e",
    "conclusion": "e52238",
    "design_reason": "ff7002",
    "limitation": "ffbf9e",
    "metric": "ffff00",
}
KIND_LABEL = {
    "keyword": "关键词",
    "keyword_note": "关键词解释",
    "definition": "概念定义",
    "related": "相关研究",
    "method": "研究方法",
    "conclusion": "研究结论",
    "design_reason": "实验设计原因",
    "limitation": "局限",
    "metric": "指标说明",
}
KIND_LIMIT = {
    "keyword": 20,
    "keyword_note": 6,
    "definition": 12,
    "related": 10,
    "method": 10,
    "conclusion": 12,
    "design_reason": 8,
    "limitation": 8,
    "metric": 14,
}
KIND_HITS = {
    "keyword": 1,
    "metric": 2,
}
NOTE_KINDS = {"keyword_note", "definition", "related", "design_reason", "limitation", "metric"}
READING_KINDS = {kind for kind in KIND_LIMIT if kind != "metric"}


@dataclass
class Glyph:
    text: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class Mark:
    kind: str
    quote: str
    note: str


@dataclass
class AnnotateResult:
    path: Path
    added: int = 0
    skipped: int = 0
    already: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def message(self) -> str:
        if self.added == 0 and self.already:
            return "这些批注已经在原文里，没有重复添加。"
        if self.added == 0:
            return "没有写入批注。模型给出的句子在 PDF 文字层里对不上，或原文没有可定位的文字。"
        parts = [f"{KIND_LABEL[kind]} {count}" for kind, count in self.by_kind.items() if count]
        detail = "，".join(parts)
        tail = f"另有 {self.skipped} 处在原文里没对上，已跳过。" if self.skipped else "再次运行不会重复添加。"
        legend = "淡黄是关键词，金色是定义，青色是相关研究，粉色是方法，红色是结论和发现，橙色是设计原因，肉色是局限。结果和表格里的指标用正黄标出，批注里写为什么用、范围、本文数值和算法。"
        return f"已在原文写入 {self.added} 处批注（{detail}）。{tail}{legend}"

    def to_dict(self) -> dict:
        return {
            "ok": True,
            "added": self.added,
            "skipped": self.skipped,
            "already": self.already,
            "by_kind": self.by_kind,
            "message": self.message(),
            "path": str(self.path),
        }


def _char_width(char: str, size: float) -> float:
    if char.isspace():
        return size * 0.3
    if ord(char) > 255:
        return size
    return size * 0.5


def _placed_text(cm, tm, font_size: float) -> tuple[float, float, float]:
    """把文字矩阵换算成页面上的起点和字号。PDF 常把缩放写在矩阵里，字号本身是 1。"""
    x = cm[0] * tm[4] + cm[2] * tm[5] + cm[4]
    y = cm[1] * tm[4] + cm[3] * tm[5] + cm[5]
    scale = math.hypot(tm[0], tm[1]) * math.hypot(cm[0], cm[1])
    size = abs(float(font_size or 0) * scale)
    return x, y, size or 10.0


def _utf16_hex(value: str) -> str:
    if len(value) % 4:
        return ""
    chars = []
    for index in range(0, len(value), 4):
        code = int(value[index : index + 4], 16)
        if code:
            chars.append(chr(code))
    return "".join(chars)


def _font_metrics(font) -> tuple[dict[int, str], dict[int, float]]:
    """字节到文字、字节到字宽（千分之 em）。没有宽度表时字宽为空，调用方用估计值。"""
    mapping: dict[int, str] = {}
    widths: dict[int, float] = {}
    try:
        if "/ToUnicode" in font:
            data = font["/ToUnicode"].get_data().decode("latin1", errors="replace")
            for block in re.finditer(r"beginbfchar(.*?)endbfchar", data, re.S):
                for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block.group(1)):
                    mapping[int(src, 16)] = _utf16_hex(dst)
            for block in re.finditer(r"beginbfrange(.*?)endbfrange", data, re.S):
                for src, end, dst in re.findall(
                    r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>",
                    block.group(1),
                ):
                    start_code = int(src, 16)
                    base = int(dst, 16)
                    for offset, code in enumerate(range(start_code, int(end, 16) + 1)):
                        mapping[code] = chr(base + offset)
        if "/Widths" in font and "/FirstChar" in font:
            first = int(font["/FirstChar"])
            for offset, width in enumerate(font["/Widths"]):
                widths[first + offset] = float(width) / 1000
    except Exception:
        return mapping, widths
    return mapping, widths


def _page_font_metrics(page) -> dict[str, tuple[dict[int, str], dict[int, float]]]:
    resources = page.get("/Resources")
    if resources is None:
        return {}
    fonts = resources.get_object().get("/Font")
    if fonts is None:
        return {}
    parsed = {}
    for name, value in fonts.get_object().items():
        try:
            parsed[str(name)] = _font_metrics(value.get_object())
        except Exception:
            continue
    return parsed


def _scales(cm, tm, font_size: float) -> tuple[float, float, float, float]:
    x = cm[0] * tm[4] + cm[2] * tm[5] + cm[4]
    y = cm[1] * tm[4] + cm[3] * tm[5] + cm[5]
    horizontal = math.hypot(tm[0], tm[1]) * math.hypot(cm[0], cm[1])
    vertical = math.hypot(tm[2], tm[3]) * math.hypot(cm[2], cm[3])
    width_size = abs(float(font_size or 0) * horizontal) or 10.0
    height = abs(float(font_size or 0) * vertical) or width_size
    return x, y, width_size, height


def _decode_codes(operand, mapping: dict[int, str]) -> list[tuple[int, str]]:
    if isinstance(operand, bytes):
        raw = operand
    else:
        raw = str(operand).encode("latin1", errors="replace")
    decoded = []
    for code in raw:
        text = mapping.get(code)
        if text is None:
            text = chr(code) if 32 <= code < 127 else ""
        if text:
            decoded.append((code, text))
    return decoded


def _show_glyphs(
    pieces: list,
    x: float,
    y: float,
    rendered: float,
    height: float,
    widths: dict[int, float],
    mapping: dict[int, str],
) -> tuple[list[Glyph], float]:
    """按字体宽度和 TJ 字距把一段文字排到页面上。正数 TJ 往左收，负数拉开。"""
    glyphs: list[Glyph] = []
    cursor = x
    for piece in pieces:
        if isinstance(piece, (bytes, str)):
            decoded = _decode_codes(piece, mapping)
        else:
            try:
                adjustment = float(piece)
            except (TypeError, ValueError):
                continue
            delta = -adjustment / 1000 * rendered
            if delta > rendered * 0.15:
                glyphs.append(Glyph(" ", cursor, y, delta, height))
            cursor += delta
            continue
        for code, text in decoded:
            ratio = widths.get(code)
            if ratio is None:
                ratio = 0.33 if text.isspace() else (1.0 if any(ord(char) > 255 for char in text) else 0.5)
            width = rendered * ratio
            share = width / len(text)
            for char in text:
                glyphs.append(Glyph(char, cursor, y, share, height))
                cursor += share
    return glyphs, cursor


def glyphs_on_page(page) -> list[Glyph]:
    positioned = _glyphs_from_operands(page)
    if positioned:
        return positioned
    return _glyphs_from_text_visitor(page)


def _glyphs_from_operands(page) -> list[Glyph]:
    fonts = _page_font_metrics(page)
    found: list[Glyph] = []
    state = {
        "mapping": {},
        "widths": {},
        "font_size": 10.0,
        "origin": None,
        "cursor": 0.0,
        "y": 0.0,
        "rendered": 10.0,
        "height": 10.0,
    }

    def visit(operator, operands, cm, tm) -> None:
        if operator == b"Tf" and operands:
            metrics = fonts.get(str(operands[0]), ({}, {}))
            state["mapping"], state["widths"] = metrics
            try:
                state["font_size"] = float(operands[1])
            except (TypeError, ValueError):
                pass
            return
        if operator not in (b"Tj", b"TJ") or not operands:
            return
        x, y, rendered, height = _scales(cm, tm, state["font_size"])
        origin = (round(x, 1), round(y, 1))
        if origin != state["origin"]:
            state["origin"] = origin
            state["cursor"] = x
            state["y"] = y
            state["rendered"] = rendered
            state["height"] = height
        pieces = list(operands[0]) if operator == b"TJ" else [operands[0]]
        glyphs, state["cursor"] = _show_glyphs(
            pieces,
            state["cursor"],
            state["y"],
            state["rendered"],
            state["height"],
            state["widths"],
            state["mapping"],
        )
        found.extend(glyphs)

    try:
        page.extract_text(visitor_operand_before=visit)
    except Exception:
        return []
    return found


def _glyphs_from_text_visitor(page) -> list[Glyph]:
    runs: list[tuple[float, float, float, str]] = []
    # 同一行里，连续的词经常被回报成同一个起点。先拼成一段，再按下一个起点收束字宽。
    cursor = {"key": None}

    def visit(text, cm, tm, font_dict, font_size) -> None:
        if not text:
            return
        x, y, size = _placed_text(cm, tm, font_size)
        key = (round(x, 1), round(y, 1), round(size, 1))
        chunk = text.replace("\n", "")
        if not chunk:
            return
        if runs and key == cursor["key"]:
            origin_x, origin_y, origin_size, previous = runs[-1]
            runs[-1] = (origin_x, origin_y, origin_size, previous + chunk)
            return
        cursor["key"] = key
        runs.append((x, y, size, chunk))

    page.extract_text(visitor_text=visit)
    return _glyphs_from_runs(runs)


def _same_baseline(y: float, other: float, size: float) -> bool:
    # 双栏常常差两三个点，不能把左右栏收成同一行。
    return abs(y - other) <= max(2.0, size * 0.22)


def _glyphs_from_runs(runs: list[tuple[float, float, float, str]]) -> list[Glyph]:
    lines: list[list[tuple[float, float, float, str]]] = []
    for run in sorted(runs, key=lambda item: (-item[1], item[0])):
        if lines and _same_baseline(run[1], lines[-1][0][1], run[2]):
            lines[-1].append(run)
        else:
            lines.append([run])
    glyphs: list[Glyph] = []
    for line in lines:
        line.sort(key=lambda item: item[0])
        for index, (x, y, size, text) in enumerate(line):
            natural = sum(_char_width(char, size) for char in text) or 1.0
            scale = 1.0
            if index + 1 < len(line):
                gap = line[index + 1][0] - x
                leftover = gap - natural
                overlaps = gap > size * 0.3 and leftover < 0
                if overlaps:
                    scale = gap * 0.98 / natural
            cursor = x
            for char in text:
                width = _char_width(char, size) * scale
                glyphs.append(Glyph(char, cursor, y, width, size))
                cursor += width
    return glyphs


def read_pdf_pages(path: Path) -> list[list[Glyph]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AutoPaperError("批注 PDF 需要安装 pypdf") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise AutoPaperError("这份 PDF 已加密，无法批注")
        return [glyphs_on_page(page) for page in reader.pages]
    except AutoPaperError:
        raise
    except Exception as exc:
        raise AutoPaperError(f"无法读取 PDF：{exc}") from exc


def _reading_lines(glyphs: list[Glyph]) -> list[list[Glyph]]:
    """按栏从上到下排。双栏同一行的左右文字不再拼成一句。"""
    rows: list[list[Glyph]] = []
    for glyph in sorted(glyphs, key=lambda item: (-item.y, item.x)):
        if rows and _same_baseline(glyph.y, rows[-1][0].y, glyph.h):
            rows[-1].append(glyph)
        else:
            rows.append([glyph])
    segments: list[list[Glyph]] = []
    for row in rows:
        row.sort(key=lambda item: item.x)
        current = [row[0]]
        for glyph in row[1:]:
            previous = current[-1]
            gap = glyph.x - (previous.x + previous.w)
            if gap > max(16.0, glyph.h * 1.4):
                segments.append(current)
                current = [glyph]
            else:
                current.append(glyph)
        segments.append(current)
    columns: list[list[list[Glyph]]] = []
    for segment in sorted(segments, key=lambda item: item[0].x):
        anchor = segment[0].x
        for column in columns:
            if abs(anchor - column[0][0].x) <= 40:
                column.append(segment)
                break
        else:
            columns.append([segment])
    columns.sort(key=lambda column: column[0][0].x)
    ordered: list[list[Glyph]] = []
    for column in columns:
        column.sort(key=lambda segment: -segment[0].y)
        ordered.extend(column)
    return ordered


def _line_chars(line: list[Glyph]) -> list[str]:
    chars: list[str] = []
    cursor = None
    for glyph in line:
        if cursor is not None and glyph.x - cursor > glyph.h * 0.18 and not (chars and chars[-1].isspace()):
            chars.append(" ")
        chars.append(glyph.text)
        cursor = glyph.x + glyph.w
    return chars


def page_plain_text(glyphs: list[Glyph]) -> str:
    if not glyphs:
        return ""
    return "\n".join("".join(_line_chars(line)).strip() for line in _reading_lines(glyphs))


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def _fold(text: str) -> str:
    """大小写折叠，但保持字符数不变，这样匹配下标还能对回字形。"""
    return "".join(char.lower() if "A" <= char <= "Z" else char for char in text)


def _page_index(glyphs: list[Glyph]) -> tuple[str, list[Glyph]]:
    chars: list[str] = []
    owners: list[Glyph] = []
    for line in _reading_lines(glyphs):
        if chars and not chars[-1].isspace():
            chars.append(" ")
            owners.append(Glyph(" ", line[0].x, line[0].y, line[0].h * 0.3, line[0].h))
        previous = None
        for glyph in line:
            gap = previous is not None and glyph.x - (previous.x + previous.w) > glyph.h * 0.18
            if gap and chars and not chars[-1].isspace() and not glyph.text.isspace():
                chars.append(" ")
                owners.append(Glyph(" ", previous.x + previous.w, glyph.y, glyph.x - (previous.x + previous.w), glyph.h))
            chars.append(glyph.text)
            owners.append(glyph)
            previous = glyph
    return "".join(chars), owners


def locate_quote(pages: list[list[Glyph]], quote: str, limit: int) -> list[tuple[int, list[Glyph]]]:
    needle = _fold(_collapsed(quote))
    if len(needle) < 2:
        return []
    found = _scan_quote(pages, needle, limit, loose=False)
    if found:
        return found
    loose = _loose_text(quote)
    if len(loose) < 2:
        return []
    return _scan_quote(pages, loose, limit, loose=True)


def _scan_quote(pages: list[list[Glyph]], needle: str, limit: int, *, loose: bool) -> list[tuple[int, list[Glyph]]]:
    found: list[tuple[int, list[Glyph]]] = []
    for page_index, glyphs in enumerate(pages):
        raw, owners = _page_index(glyphs)
        haystack, owners = _dehyphen(raw, owners) if loose else (_fold(raw), owners)
        start = 0
        while len(found) < limit:
            at = haystack.find(needle, start)
            if at < 0:
                break
            found.append((page_index, owners[at : at + len(needle)]))
            start = at + max(len(needle), 1)
        if len(found) >= limit:
            break
    return found


_PUNCT = str.maketrans({"–": "-", "—": "-", "−": "-", "’": "'", "‘": "'", "“": '"', "”": '"'})


def _loose_text(quote: str) -> str:
    chars, _owners = _dehyphen(quote, [Glyph("", 0, 0, 0, 0) for _ in quote])
    return chars


def _dehyphen(text: str, owners: list[Glyph]) -> tuple[str, list[Glyph]]:
    """去掉断行连字符，并把弯引号折成直引号，方便对上模型抄出来的句子。"""
    chars: list[str] = []
    mapped: list[Glyph] = []
    index = 0
    size = len(text)
    while index < size:
        char = text[index].translate(_PUNCT)
        if char == "\u00ad":
            index += 1
            continue
        if char == "-" and _hyphen_between_letters(text, index):
            index += 1
            while index < size and text[index].isspace():
                index += 1
            continue
        if char.isspace():
            if chars and not chars[-1].isspace():
                chars.append(" ")
                mapped.append(owners[index])
            index += 1
            continue
        chars.append(char.lower() if "A" <= char <= "Z" else char)
        mapped.append(owners[index])
        index += 1
    if chars and chars[-1] == " ":
        chars.pop()
        mapped.pop()
    return "".join(chars), mapped


def _hyphen_between_letters(text: str, index: int) -> bool:
    previous = index - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    if previous < 0 or not text[previous].isalpha():
        return False
    following = index + 1
    while following < len(text) and text[following].isspace():
        following += 1
    return following < len(text) and text[following].isalpha()


def _quads(glyphs: list[Glyph]) -> tuple[tuple[float, float, float, float], list[tuple[float, ...]]]:
    unique: list[Glyph] = []
    seen: set[int] = set()
    for glyph in glyphs:
        if id(glyph) in seen or not glyph.text.strip():
            continue
        seen.add(id(glyph))
        unique.append(glyph)
    if not unique:
        unique = [glyph for glyph in glyphs[:1]]
    lines: list[list[Glyph]] = []
    for glyph in sorted(unique, key=lambda item: (-item.y, item.x)):
        if lines and _same_baseline(glyph.y, lines[-1][0].y, glyph.h):
            lines[-1].append(glyph)
        else:
            lines.append([glyph])
    quads = []
    left = top = right = bottom = None
    for line in lines:
        x0 = min(item.x for item in line)
        x1 = max(item.x + item.w for item in line)
        y0 = min(item.y - item.h * 0.25 for item in line)
        y1 = max(item.y + item.h * 0.85 for item in line)
        # 阅读器按 左上、右上、左下、右下 解释四个顶点。
        quads.append((x0, y1, x1, y1, x0, y0, x1, y0))
        left = x0 if left is None else min(left, x0)
        right = x1 if right is None else max(right, x1)
        bottom = y0 if bottom is None else min(bottom, y0)
        top = y1 if top is None else max(top, y1)
    return (left or 0, bottom or 0, right or 0, top or 0), quads


def _mark_name(kind: str, quote: str, index: int) -> str:
    digest = hashlib.sha1(f"{kind}|{_fold(_collapsed(quote))}|{index}".encode("utf-8")).hexdigest()[:16]
    return f"autopaper-{digest}"


def _drop_autopaper_annotations(page) -> None:
    """重跑时拿掉本程序以前写的批注，好让新的位置盖上旧的偏移。"""
    from pypdf.generic import ArrayObject, NameObject

    annots = page.get("/Annots")
    if not annots:
        return
    kept = []
    for item in annots:
        name = str(item.get_object().get("/NM") or "")
        if not name.startswith("autopaper-"):
            kept.append(item)
    if len(kept) == len(annots):
        return
    if kept:
        page[NameObject("/Annots")] = ArrayObject(kept)
    else:
        del page[NameObject("/Annots")]


def _existing_names(page) -> set[str]:
    names: set[str] = set()
    annots = page.get("/Annots")
    if not annots:
        return names
    for item in annots:
        obj = item.get_object()
        name = obj.get("/NM")
        if name:
            names.add(str(name))
    return names


def _contents(mark: Mark) -> str:
    label = KIND_LABEL[mark.kind]
    note = mark.note.strip()
    if note:
        return f"{label}\n{note}"
    return label


def parse_marks(payload, *, allow: set[str] | None = None) -> list[Mark]:
    if isinstance(payload, dict):
        raw = payload.get("items")
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = None
    if not isinstance(raw, list):
        raise AutoPaperError("模型没有返回批注列表")
    counts = {kind: 0 for kind in KIND_LIMIT}
    seen: set[str] = set()
    marks: list[Mark] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        quote = " ".join(str(item.get("quote") or "").split())
        quote_limit = 80 if kind == "keyword" else 420
        if kind not in KIND_LIMIT or (allow is not None and kind not in allow):
            continue
        if len(quote) < 2 or len(quote) > quote_limit:
            continue
        if counts[kind] >= KIND_LIMIT[kind]:
            continue
        key = _fold(quote)
        if key in seen:
            continue
        seen.add(key)
        note = _metric_note(item) if kind == "metric" else str(item.get("note") or "").strip()
        if kind in NOTE_KINDS and not note:
            note = "论文文本未说明"
        counts[kind] += 1
        marks.append(Mark(kind, quote, note[:1800]))
    return marks


def _metric_note(item: dict) -> str:
    why = _filled(item.get("why"))
    span = _filled(item.get("range"))
    finding = _filled(item.get("finding"))
    calculation = _filled(item.get("calculation"))
    measures = _filled(item.get("measures"))
    return f"为什么用：{why}\n范围：{span}\n本文：{finding}\n计算：{calculation}\n衡量：{measures}"


def _filled(value) -> str:
    text = " ".join(str(value or "").split())
    return text or "论文文本未说明"


def write_marks(path: Path, marks: list[Mark]) -> AnnotateResult:
    from pypdf import PdfReader, PdfWriter
    from pypdf.annotations import Highlight, Text
    from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject

    pages = read_pdf_pages(path)
    if not any(pages):
        raise AutoPaperError("这份 PDF 没有可定位的文字，扫描件无法自动批注")
    result = AnnotateResult(path=path)
    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.append(reader)
    for page in writer.pages:
        _drop_autopaper_annotations(page)
    known = [_existing_names(page) for page in writer.pages]
    for mark in marks:
        hits = locate_quote(pages, mark.quote, KIND_HITS.get(mark.kind, 1))
        if not hits:
            result.skipped += 1
            continue
        for index, (page_index, glyphs) in enumerate(hits):
            name = _mark_name(mark.kind, mark.quote, index)
            if name in known[page_index]:
                result.already += 1
                continue
            rect, quads = _quads(glyphs)
            points = ArrayObject()
            for quad in quads:
                for value in quad:
                    points.append(FloatObject(value))
            color = KIND_COLOR[mark.kind]
            highlight = Highlight(
                rect=rect,
                quad_points=points,
                highlight_color=color,
                printing=True,
                title_bar="AutoPaper",
            )
            highlight[NameObject("/Contents")] = TextStringObject(_contents(mark))
            highlight[NameObject("/NM")] = TextStringObject(name)
            writer.add_annotation(page_index, highlight)
            known[page_index].add(name)
            if mark.kind in NOTE_KINDS:
                note_name = name + "-note"
                if note_name not in known[page_index]:
                    x0, _y0, _x1, y1 = rect
                    note = Text(
                        rect=(x0, y1 + 2, x0 + 18, y1 + 20),
                        text=_contents(mark),
                        flags=4,
                        title_bar="AutoPaper",
                    )
                    note[NameObject("/NM")] = TextStringObject(note_name)
                    note[NameObject("/Name")] = NameObject("/Comment")
                    note[NameObject("/C")] = ArrayObject(
                        [FloatObject(int(color[i : i + 2], 16) / 255) for i in (0, 2, 4)]
                    )
                    writer.add_annotation(page_index, note)
                    known[page_index].add(note_name)
            result.added += 1
            result.by_kind[mark.kind] = result.by_kind.get(mark.kind, 0) + 1
    if result.added:
        _replace_pdf(path, writer)
    return result


def _replace_pdf(path: Path, writer) -> None:
    temporary = None
    try:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=path.parent)
        temporary = Path(handle.name)
        writer.write(handle)
        handle.close()
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise AutoPaperError("原文正被阅读器占用，请先关闭后再批注。") from exc
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def annotate_pdf(path: Path, llm) -> AnnotateResult:
    pages = read_pdf_pages(path)
    plain = "\n\n".join(page_plain_text(page) for page in pages).strip()
    if not plain:
        raise AutoPaperError("这份 PDF 没有可定位的文字，扫描件无法自动批注")
    clipped = clip_paper_text(plain)
    reading = llm.complete_json(
        ANNOTATE_SYSTEM,
        annotate_user_prompt(clipped),
        expect="object",
        max_tokens=6144,
    )
    metrics = llm.complete_json(
        ANNOTATE_SYSTEM,
        annotate_metric_prompt(clipped),
        expect="object",
        max_tokens=6144,
    )
    marks = parse_marks(reading, allow=READING_KINDS) + parse_marks(metrics, allow={"metric"})
    return write_marks(path, marks)
