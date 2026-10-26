from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from autopaper.analysis.pdfnotes import Glyph, Mark, locate_quote, parse_marks, write_marks


def _write_pdf(path: Path, lines: list[str]) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")
    page = writer.pages[0]
    resources = DictionaryObject()
    resources[NameObject("/Font")] = DictionaryObject({NameObject("/F1"): font})
    page[NameObject("/Resources")] = resources
    commands = ["BT", "/F1 12 Tf"]
    y = 700
    for line in lines:
        safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append(f"1 0 0 1 72 {y} Tm ({safe}) Tj")
        y -= 18
    commands.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


def _annotations(path: Path) -> list:
    reader = PdfReader(str(path))
    found = []
    for page in reader.pages:
        for item in page.get("/Annots") or []:
            found.append(item.get_object())
    return found


def _color(annotation) -> tuple[float, float, float]:
    return tuple(round(float(channel), 3) for channel in annotation["/C"])


def test_parse_marks_formats_metric_notes_and_drops_invalid_items():
    marks = parse_marks(
        {
            "items": [
                {"kind": "keyword", "quote": "grant success"},
                {"kind": "keyword", "quote": "Grant Success"},
                {"kind": "definition", "quote": "grant success"},
                {"kind": "nope", "quote": "grant success"},
                {"kind": "keyword", "quote": "x"},
                {"kind": "keyword", "quote": "k" * 81},
                {
                    "kind": "definition",
                    "quote": "Attitudes are overall evaluations of objects",
                    "note": "态度是对对象的总体评价。",
                },
                {
                    "kind": "limitation",
                    "quote": "Measurement invariance was not assessed",
                    "note": "没有检验测量不变性。",
                },
                {
                    "kind": "metric",
                    "quote": "h-index",
                    "why": "用来概括学者的持续被引",
                    "range": "通用标准：h 为非负整数，没有固定上限。",
                    "finding": "论文文本未说明",
                    "calculation": "通用定义：取 h 使有 h 篇论文至少被引 h 次",
                    "measures": "学者的持续影响力",
                },
                {"kind": "metric", "quote": "RMSEA"},
                {"kind": "related", "quote": "Smith 2019 showed the same pattern"},
            ]
        }
    )
    assert [mark.kind for mark in marks] == ["keyword", "definition", "limitation", "metric", "metric", "related"]
    assert marks[1].note == "态度是对对象的总体评价。"
    assert marks[2].note == "没有检验测量不变性。"
    metric = marks[3]
    assert metric.note.startswith("为什么用：")
    assert "范围：通用标准：" in metric.note
    assert "本文：论文文本未说明" in metric.note
    assert "计算：通用定义：" in metric.note
    assert "衡量：学者的持续影响力" in metric.note
    assert marks[4].note.split("\n") == [
        "为什么用：论文文本未说明",
        "范围：论文文本未说明",
        "本文：论文文本未说明",
        "计算：论文文本未说明",
        "衡量：论文文本未说明",
    ]
    assert marks[5].note == "论文文本未说明"
    only_metrics = parse_marks(
        {"items": [{"kind": "keyword", "quote": "grant success"}, {"kind": "metric", "quote": "CFI"}]},
        allow={"metric"},
    )
    assert [mark.kind for mark in only_metrics] == ["metric"]


def test_locate_quote_joins_lines_and_keeps_the_page():
    top = [Glyph(char, 72 + index * 6, 700, 6, 12) for index, char in enumerate("Alpha beta")]
    bottom = [Glyph(char, 72 + index * 6, 680, 6, 12) for index, char in enumerate("gamma delta")]
    hits = locate_quote([top + bottom], "beta gamma", 1)
    assert len(hits) == 1
    page_index, glyphs = hits[0]
    assert page_index == 0
    assert "".join(glyph.text for glyph in glyphs) == "beta gamma"


def test_write_marks_colors_quotes_and_does_not_duplicate(tmp_path: Path):
    path = tmp_path / "paper.pdf"
    _write_pdf(
        path,
        [
            "promotional language predicts grant success",
            "We measured outcomes with the h-index",
        ],
    )
    marks = [
        Mark("keyword", "promotional language", ""),
        Mark("conclusion", "grant success", ""),
        Mark("metric", "h-index", "为什么用：概括持续被引\n范围：通用标准：非负整数\n本文：论文文本未说明\n计算：通用定义：至少被引 h 次\n衡量：持续影响力"),
        Mark("method", "this sentence is not in the pdf", ""),
    ]
    result = write_marks(path, marks)
    assert result.added == 3
    assert result.skipped == 1
    annotations = _annotations(path)
    highlights = [item for item in annotations if str(item["/Subtype"]) == "/Highlight"]
    notes = [item for item in annotations if str(item["/Subtype"]) == "/Text"]
    colors = {_color(item) for item in highlights}
    assert (1.0, 1.0, 0.0) in colors
    assert (round(0xE5 / 255, 3), round(0x22 / 255, 3), round(0x38 / 255, 3)) in colors
    assert (round(0xFD / 255, 3), round(0xF4 / 255, 3), round(0x85 / 255, 3)) in colors
    assert len(notes) == 1
    assert "为什么用" in str(notes[0]["/Contents"])
    assert "通用定义" in str(notes[0]["/Contents"])
    assert "通用标准" in str(notes[0]["/Contents"])

    again = write_marks(path, marks)
    assert again.added == 3
    assert len(_annotations(path)) == len(annotations)


def test_annotate_pdf_merges_reading_marks_with_result_metrics(tmp_path: Path):
    from autopaper.analysis.pdfnotes import annotate_pdf

    path = tmp_path / "paper.pdf"
    _write_pdf(path, ["CFI = .97 shows excellent fit", "We used confirmatory factor analysis"])

    class Fake:
        def complete_json(self, system, user, **kwargs):
            if "不要输出 metric" in user:
                return {
                    "items": [
                        {"kind": "keyword", "quote": "confirmatory factor analysis"},
                        {"kind": "metric", "quote": "CFI = .97", "why": "不该来自第一轮"},
                        {"kind": "conclusion", "quote": "excellent fit"},
                    ]
                }
            return {
                "items": [
                    {
                        "kind": "metric",
                        "quote": "CFI = .97",
                        "why": "检验拟合",
                        "range": "通用标准：0 到 1",
                        "finding": "论文报告 CFI = .97。通常意味着：拟合良好。",
                        "calculation": "通用定义：比较模型",
                        "measures": "拟合",
                    }
                ]
            }

    result = annotate_pdf(path, Fake())
    assert result.by_kind == {"keyword": 1, "conclusion": 1, "metric": 1}
    notes = [item for item in _annotations(path) if str(item["/Subtype"]) == "/Text"]
    text = str(notes[0]["/Contents"])
    assert "检验拟合" in text
    assert "通常意味着" in text
    assert "不该来自第一轮" not in text


def test_locate_quote_joins_a_word_split_by_a_hyphen():
    pieces = [Glyph(char, 72 + index * 6, 700, 6, 12) for index, char in enumerate("va-")]
    rest = [Glyph(char, 72 + index * 6, 680, 6, 12) for index, char in enumerate("lidity checks")]
    hits = locate_quote([pieces + rest], "validity checks", 1)
    assert len(hits) == 1
    assert "".join(glyph.text for glyph in hits[0][1] if glyph.text.strip()) == "validitychecks"


def test_show_glyphs_follow_font_widths_and_word_gaps():
    from autopaper.analysis.pdfnotes import _show_glyphs

    widths = {ord("i"): 0.2, ord("m"): 0.9}
    glyphs, cursor = _show_glyphs(
        ["i", -250, "m"],
        x=100,
        y=200,
        rendered=10,
        height=10,
        widths=widths,
        mapping={},
    )
    assert [glyph.text for glyph in glyphs] == ["i", " ", "m"]
    assert glyphs[0].w == 2
    assert glyphs[2].x == 104.5
    assert cursor == 113.5
