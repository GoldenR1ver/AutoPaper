"""从论文原文识别年份、期刊或会议、DOI 和链接。不调用模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from autopaper.analysis.reader import paper_meta
from autopaper.models import Paper
from autopaper.storage.db import KnowledgeBase
from autopaper.textutil import collapse_ws, normalize_doi

MAX_YEAR = datetime.now().year + 1
PLACEHOLDER_DOIS = {
    "10.1145/1122445.1122456",
    "10.1145/nnnnnnn.nnnnnnn",
}
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
VENUE_PATTERNS = (
    (re.compile(r"JAMA\s+Network\s+Open", re.IGNORECASE), "JAMA Network Open"),
    (re.compile(r"Proceedings of the National Academy of Sciences|\bPNAS\b", re.IGNORECASE), "PNAS"),
    (re.compile(r"Cooperative and Human Aspects of Software Engineering", re.IGNORECASE), "CHASE"),
    (re.compile(r"Software Engineering in Society", re.IGNORECASE), "ICSE-SEIS"),
    (re.compile(r"International Conference on Software Engineering", re.IGNORECASE), "ICSE"),
    (re.compile(r"Computer Supported Cooperative Work", re.IGNORECASE), "CSCW"),
    (re.compile(r"Human Factors in Computing Systems", re.IGNORECASE), "CHI"),
)
DOI_VENUES = (
    (re.compile(r"^10\.1073/pnas", re.IGNORECASE), "PNAS"),
    (re.compile(r"^10\.1001/jamanetworkopen", re.IGNORECASE), "JAMA Network Open"),
    (re.compile(r"^10\.1109/chase", re.IGNORECASE), "CHASE"),
    (re.compile(r"^10\.1109/icse", re.IGNORECASE), "ICSE"),
)


@dataclass
class Bibliography:
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""

    def as_dict(self) -> dict:
        return {"year": self.year, "venue": self.venue, "doi": self.doi, "url": self.url}


def extract_bibliography(path: Path) -> Bibliography:
    """读取本地 PDF、TXT 或 Markdown，返回识别到的题录。识别不到的字段留空。"""
    if not path or not Path(path).exists():
        return Bibliography()
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        meta, text = _pdf_parts(path)
    elif suffix in {".txt", ".md", ".markdown"}:
        meta, text = "", path.read_text(encoding="utf-8", errors="replace")[:20000]
    else:
        return Bibliography()
    return extract_from_parts(path.stem, meta, text)


def extract_from_parts(filename: str, metadata: str, text: str) -> Bibliography:
    page = text[:12000]
    doi = _choose_doi(page) or _choose_doi(metadata)
    proceedings_venue, proceedings_year = _proceedings(page)
    venue = _venue_from_doi(doi) or proceedings_venue or _venue_from_text(page[:2000]) or _venue_from_name(filename)
    year = _year_from_doi(doi) or proceedings_year or _copyright_year(page) or _year_from_name(filename)
    url = f"https://doi.org/{doi}" if doi else _first_url(page)
    return Bibliography(year=year, venue=venue, doi=doi, url=url)


def fill_paper(kb: KnowledgeBase, paper: Paper, *, only_missing: bool = True) -> dict:
    """用原文识别结果补论文题录。only_missing 时不覆盖已经写过的字段。"""
    found = extract_bibliography(Path(paper.local_path)) if paper.local_path else Bibliography()
    filled: list[str] = []
    if found.year and (paper.year is None or not only_missing) and paper.year != found.year:
        paper.year = found.year
        filled.append("year")
    if found.venue and (not paper.venue or not only_missing) and paper.venue != found.venue:
        paper.venue = found.venue
        filled.append("venue")
    if found.doi and (not paper.doi or not only_missing) and paper.doi != found.doi:
        paper.doi = found.doi
        filled.append("doi")
    url = found.url
    if url and (not paper.url or not only_missing) and paper.url != url:
        paper.url = url
        filled.append("url")
    if filled:
        kb.update_bibliography(paper)
        refresh_note_header(kb, paper)
    return _public_record(paper, found, filled)


def set_bibliography(
    kb: KnowledgeBase,
    paper: Paper,
    *,
    year: int | None,
    venue: str,
    doi: str,
    url: str,
) -> dict:
    """按用户输入改题录。链接留空且写了 DOI 时，链接写成 doi.org。"""
    paper.year = year
    paper.venue = collapse_ws(venue) or None
    paper.doi = normalize_doi(doi)
    paper.url = collapse_ws(url) or None
    if paper.doi and not paper.url:
        paper.url = f"https://doi.org/{paper.doi}"
    kb.update_bibliography(paper)
    refresh_note_header(kb, paper)
    return _public_record(paper, Bibliography(paper.year, paper.venue or "", paper.doi or "", paper.url or ""), ["year", "venue", "doi", "url"])


def fill_local_papers(kb: KnowledgeBase) -> list[dict]:
    return [fill_paper(kb, paper, only_missing=True) for paper in kb.list_local_papers()]


def recognize_uploaded(kb: KnowledgeBase, papers: list[Paper]) -> tuple[list[Paper], list[dict]]:
    """新上传的论文字段是空的，把原文识别结果写进去，并返回识别值。"""
    records = []
    fresh: list[Paper] = []
    for paper in papers:
        records.append(fill_paper(kb, paper, only_missing=True))
        loaded = kb.resolve_paper(paper.key)
        if loaded is not None:
            fresh.append(loaded)
    return fresh, records


def format_recognized(record: dict) -> str:
    found = record.get("recognized") or {}
    return "原文识别：年份 {year}，期刊或会议 {venue}，DOI {doi}，链接 {url}".format(
        year=found.get("year") or "未识别",
        venue=found.get("venue") or "未识别",
        doi=found.get("doi") or "未识别",
        url=found.get("url") or "未识别",
    )


def refresh_note_header(kb: KnowledgeBase, paper: Paper) -> None:
    reading = kb.get_reading(paper.key)
    if not reading or not reading.get("markdown"):
        return
    updated = replace_meta_block(reading["markdown"], paper)
    if updated == reading["markdown"]:
        return
    kb.replace_reading_markdown(paper.key, updated)
    if not paper.local_path:
        return
    note = Path(paper.local_path).parent / "note.md"
    if note.exists():
        note.write_text(updated if updated.endswith("\n") else updated + "\n", encoding="utf-8")


def replace_meta_block(markdown: str, paper: Paper) -> str:
    lines = markdown.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith("标题：")), None)
    if start is None:
        return markdown
    end = start
    prefixes = ("标题：", "作者：", "年份：", "期刊或会议：", "DOI：", "链接：")
    while end < len(lines) and lines[end].startswith(prefixes):
        end += 1
        if lines[end - 1].startswith("链接："):
            break
    rebuilt = lines[:start] + paper_meta(paper).splitlines() + lines[end:]
    text = "\n".join(rebuilt)
    if markdown.endswith("\n"):
        text += "\n"
    return text


def _public_record(paper: Paper, found: Bibliography, filled: list[str]) -> dict:
    return {
        "key": paper.key,
        "title": paper.title,
        "year": paper.year,
        "venue": paper.venue or "",
        "doi": paper.doi or "",
        "url": paper.url or "",
        "recognized": found.as_dict(),
        "filled": filled,
    }


def _pdf_parts(path: Path) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    meta_bits = []
    if reader.metadata:
        for attr in ("title", "subject", "keywords"):
            value = getattr(reader.metadata, attr, None)
            if value:
                meta_bits.append(str(value))
    page = (reader.pages[0].extract_text() or "") if reader.pages else ""
    return "\n".join(meta_bits), page


def _choose_doi(text: str) -> str:
    if not text:
        return ""
    found = []
    for match in DOI_RE.finditer(text):
        doi = _clean_doi(match.group(0))
        if doi and doi not in found:
            found.append(doi)
    for match in re.finditer(r"doi\.org/(\S+)", text, re.IGNORECASE):
        doi = _clean_doi(match.group(1))
        if doi and doi not in found:
            found.insert(0, doi)
    found = [
        doi
        for doi in found
        if not any(other != doi and other.startswith(doi) and len(other) > len(doi) for other in found)
    ]
    return found[0] if found else ""


def _clean_doi(raw: str) -> str:
    text = raw.strip().rstrip(").,;]}")
    while text.endswith(")") and text.count("(") < text.count(")"):
        text = text[:-1]
    doi = normalize_doi(text) or ""
    if not doi or doi in PLACEHOLDER_DOIS or "nnnn" in doi:
        return ""
    suffix = doi.split("/", 1)[-1]
    if not re.search(r"\d", suffix):
        return ""
    if re.fullmatch(r"[a-z]{1,8}\.\d{4}", suffix, flags=re.IGNORECASE):
        return ""
    if re.fullmatch(r"[n0x._-]+", suffix, flags=re.IGNORECASE):
        return ""
    if len(doi) < 12:
        return ""
    return doi


def _year_from_doi(doi: str) -> int | None:
    if not doi:
        return None
    match = re.search(r"\.(19\d{2}|20\d{2})\.\d", doi)
    if not match:
        return None
    year = int(match.group(1))
    if 1990 <= year <= MAX_YEAR:
        return year
    return None


def _venue_from_doi(doi: str) -> str:
    for pattern, name in DOI_VENUES:
        if pattern.search(doi or ""):
            return name
    return ""


def _venue_from_text(text: str) -> str:
    for pattern, name in VENUE_PATTERNS:
        if pattern.search(text):
            return name
    return ""


def _proceedings(text: str) -> tuple[str, int | None]:
    matches = list(
        re.finditer(
            r"\b(CHASE|CSCW|ICSE-SEIS|SEIS|ICSE|CHI|PNAS|FSE)\s*['\u2019]?\s*((?:19|20)\d{2}|\d{2})\b",
            text,
            re.IGNORECASE,
        )
    )
    if not matches:
        return "", None
    anchor = _footer_anchor(text)
    near = [item for item in matches if anchor is not None and abs(item.start() - anchor) < 160]
    chosen = near[0] if near else matches[0]
    venue = _canonical_venue(chosen.group(1))
    raw = chosen.group(2)
    if len(raw) == 4:
        year = int(raw)
    else:
        number = int(raw)
        year = 1900 + number if number >= 90 else 2000 + number
    if not 1990 <= year <= MAX_YEAR:
        year = None
    return venue, year


def _canonical_venue(token: str) -> str:
    name = token.upper()
    if name == "SEIS":
        return "ICSE-SEIS"
    return name


def _footer_anchor(text: str) -> int | None:
    lowered = text.lower()
    for token in ("©", "copyright", "isbn", "doi.org"):
        index = lowered.find(token)
        if index != -1:
            return index
    return None


def _venue_from_name(filename: str) -> str:
    lowered = filename.casefold()
    if "cscw" in lowered:
        return "CSCW"
    if "seis" in lowered:
        return "ICSE-SEIS"
    if re.search(r"(?<![a-z])icse(?![a-z])", lowered):
        return "ICSE"
    return ""


def _copyright_year(text: str) -> int | None:
    years = [
        int(match.group(1))
        for match in re.finditer(r"(?:©|\(c\)|copyright)\s*(19\d{2}|20\d{2})", text[:8000], re.IGNORECASE)
    ]
    years = [year for year in years if 1990 <= year <= MAX_YEAR]
    return max(years) if years else None


def _year_from_name(filename: str) -> int | None:
    years = [int(item) for item in YEAR_RE.findall(filename)]
    years = [year for year in years if 1990 <= year <= MAX_YEAR]
    if len(set(years)) == 1:
        return years[0]
    match = re.search(r"(?:cscw|seis|icse|chi|chase)(\d{2})(?!\d)", filename, re.IGNORECASE)
    if not match:
        return None
    year = 2000 + int(match.group(1))
    if 1990 <= year <= MAX_YEAR:
        return year
    return None


def _first_url(text: str) -> str:
    linked = re.search(r"https?://arxiv\.org/abs/(\d{4}\.\d{4,5})", text, re.IGNORECASE)
    if linked:
        return f"https://arxiv.org/abs/{linked.group(1)}"
    stamp = re.search(r"arxiv:\s*(\d{4}\.\d{4,5})", text, re.IGNORECASE)
    if stamp:
        return f"https://arxiv.org/abs/{stamp.group(1)}"
    return ""
