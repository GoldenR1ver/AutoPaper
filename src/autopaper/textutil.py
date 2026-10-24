from __future__ import annotations

import html
import re
import unicodedata


def collapse_ws(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_match(text: str | None) -> str:
    """用于关键词匹配的文本：统一宽度、大小写和空白。"""
    return collapse_ws(unicodedata.normalize("NFKC", text or "").casefold())


def normalize_title(title: str | None) -> str:
    """用于去重的标题。标点换成空格，避免 Time-Series 和 Time Series 被拆开。"""
    text = unicodedata.normalize("NFKC", title or "").casefold()
    text = text.replace("’", "'").replace("“", " ").replace("”", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    text = collapse_ws(doi)
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = text.strip().strip(".")
    return text.casefold() or None


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    text = collapse_ws(value)
    text = re.sub(r"^arxiv:", "", text, flags=re.IGNORECASE)
    modern = re.search(r"(\d{4}\.\d{4,5})", text)
    if modern:
        return modern.group(1)
    legacy = re.search(r"([a-z\-]+/\d{7})", text, flags=re.IGNORECASE)
    if legacy:
        return legacy.group(1)
    return None


def normalize_pmid(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else None


def normalize_orcid(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", value, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def openalex_short_id(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\b([WAICT]\d+)\b", value)
    if match:
        return match.group(1)
    tail = value.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def strip_markup(text: str | None) -> str:
    if not text:
        return ""
    plain = re.sub(r"<[^>]+>", " ", text)
    return collapse_ws(html.unescape(plain))


def reconstruct_inverted_index(index: dict | str | None) -> str:
    """把 OpenAlex 的 abstract_inverted_index 还原成摘要。"""
    if not index:
        return ""
    if isinstance(index, str):
        return collapse_ws(index)
    positions: list[tuple[int, str]] = []
    for word, locations in index.items():
        if not isinstance(locations, list):
            continue
        for location in locations:
            try:
                positions.append((int(location), str(word)))
            except (TypeError, ValueError):
                continue
    positions.sort(key=lambda item: item[0])
    return collapse_ws(" ".join(word for _, word in positions))


def slugify(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").strip().casefold()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^\w\-]+", "", value, flags=re.UNICODE)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:80] or "item"


def dedupe_names(values: list[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = collapse_ws(value)
        if not text:
            continue
        marker = normalize_match(text)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(text)
    return result


def max_optional(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def like_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    return f"%{escaped}%"


def first_affiliation(value) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return collapse_ws(value) or None
    if isinstance(value, dict):
        return collapse_ws(value.get("display_name") or value.get("name") or "") or None
    if isinstance(value, list):
        for item in value:
            found = first_affiliation(item)
            if found:
                return found
    return None
