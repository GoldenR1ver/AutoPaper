from __future__ import annotations

import re

from autopaper.textutil import collapse_ws, normalize_match

_AFFILIATION = re.compile(
    r"university|institute|college|school|department|laboratory|laboratories|hospital|"
    r"foundation|academy|centre|center|gmbh|\binc\b|\bltd\b|club|company|"
    r"公司|大学|学院|研究所|医院|实验室",
    re.I,
)
_STOP = {
    "the", "a", "an", "of", "to", "for", "and", "in", "on", "with", "from",
    "toward", "towards", "using", "via", "between", "among", "into", "over",
}
_PARTICLE = {"de", "da", "di", "van", "von", "der", "la", "le", "del", "st", "bin", "al"}
_NOT_NAME = {
    "abstract", "introduction", "keywords", "index terms", "references",
    "conclusion", "conclusions", "background", "summary", "摘要", "结论", "关键词",
}
_DEGREE = re.compile(r"\b(?:ph\.?d\.?|m\.?d\.?|msc|bsc|prof\.?|dr\.?)\b", re.I)
_CORR = re.compile(
    r"(?i)(?:corresponding authors?|通信作者|通讯作者)\s*[:：]?\s*([^\n,;(]{2,80})"
)
_MARK = re.compile(r"[*†‡✉∗§]")
_ACM = re.compile(r"ACM Reference Format:\s*(.{10,900}?)\.\s*(?:19|20)\d{2}\b", re.S)


def parse_authors(front: str, full_text: str = "", metadata_author: str = "") -> tuple[tuple[str, ...], str]:
    listed = _acm_authors(full_text) or _line_authors(front)
    metadata = [_clean_person(part) for part in re.split(r"\s*(?:,|;|，|；|、|\band\b)\s*", _prepare(metadata_author or ""))]
    for name in metadata:
        if _is_person(name):
            listed.append(name)
    authors = _dedupe(listed)
    corresponding = _labeled_corresponding(full_text, authors) or _starred(front, authors)
    if corresponding and all(not _same(corresponding, name) for name in authors):
        authors.append(corresponding)
    return tuple(authors), corresponding


def parse_keywords(section: str) -> tuple[str, ...]:
    text = collapse_ws(section)
    text = re.split(r"\b(?:I{1,3}\.|[1-9]\d?\.\s+|INTRODUCTION|Introduction|摘要|引言)\b", text)[0]
    found: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,，;；、]", text):
        keyword = collapse_ws(part).strip(" .:-—–")
        marker = normalize_match(keyword)
        if not marker or marker in seen or marker in _NOT_NAME:
            continue
        words = keyword.split()
        if len(keyword) < 2 or len(keyword) > 60 or len(words) > 6:
            continue
        if re.search(r"\[|\]|\b(?:however|therefore|for example|we|this)\b", keyword, re.I):
            continue
        if any(len(word) > 30 for word in words):
            continue
        seen.add(marker)
        found.append(keyword)
        if len(found) >= 12:
            break
    return tuple(found)


def _acm_authors(text: str) -> list[str]:
    match = _ACM.search(text or "")
    if not match:
        return []
    chunk = collapse_ws(match.group(1))
    chunk = re.sub(r",?\s+and\s+", ", ", chunk, flags=re.I)
    return [name for name in (_clean_person(part) for part in chunk.split(",")) if _is_person(name)]


def _line_authors(front: str) -> list[str]:
    rows: list[tuple[str, str]] = []
    for raw in (front or "").splitlines():
        line = collapse_ws(raw)
        if not line:
            continue
        if "@" in line:
            kind = "email"
        elif _AFFILIATION.search(line):
            kind = "affil"
        elif _people_in(line):
            kind = "name"
        else:
            kind = "other"
        rows.append((kind, line))
    found: list[str] = []
    started = False
    for index, (kind, line) in enumerate(rows):
        people = _people_in(line) if kind == "name" else []
        if len(people) >= 2 or (len(people) == 1 and _followed_by_contact(rows, index)):
            found.extend(people)
            started = True
        elif started and kind == "other" and len(line) > 40:
            break
        if len(found) >= 12:
            break
    return found


def _followed_by_contact(rows: list[tuple[str, str]], index: int) -> bool:
    for kind, _line in rows[index + 1 : index + 4]:
        if kind == "name":
            return False
        if kind in {"affil", "email"}:
            return True
    return False


def _people_in(line: str) -> list[str]:
    prepared = _prepare(_strip_line_superscripts(line))
    if re.search(r"[,;，；、]|\band\b", prepared, re.I):
        pieces = re.split(r"\s*(?:,|;|，|；|、|\band\b)\s*", prepared, flags=re.I)
    else:
        pieces = [prepared]
    return [name for name in (_clean_person(piece) for piece in pieces) if _is_person(name)]


def _strip_line_superscripts(line: str) -> str:
    if not re.search(r"\d", line or ""):
        return line
    line = re.sub(r"(?<=[a-z])[a-d](?=,[a-d]|,?\d)", "", line)
    return re.sub(r"(?<=[a-z])[a-d](?=\s*,)", "", line)


def _labeled_corresponding(text: str, authors: list[str]) -> str:
    match = _CORR.search(text or "")
    if not match:
        return ""
    candidate = _clean_person(_prepare(match.group(1)))
    folded = normalize_match(candidate)
    for name in authors:
        if _same(name, candidate) or normalize_match(name) in folded:
            return name
    return ""


def _starred(front: str, authors: list[str]) -> str:
    compact = re.sub(r"\s+", "", front or "")
    for name in authors:
        spaced = re.escape(name) + r"\s*[*†‡✉∗]"
        glued = re.escape(re.sub(r"\s+", "", name)) + r"[*†‡✉∗]"
        if re.search(spaced, front or "") or re.search(glued, compact):
            return name
    return ""


def _prepare(text: str) -> str:
    return _split_glued(_DEGREE.sub(" ", text or ""))


def _split_glued(text: str) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text or "")
    return re.sub(r"(?<=\.)(?=[A-Z])", " ", text)


def _clean_person(raw: str) -> str:
    text = _DEGREE.sub("", raw or "")
    text = _MARK.sub("", text)
    text = re.sub(r"\d+", "", text)
    return collapse_ws(text).strip(" ,;.")


def _is_person(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 60 or "@" in name:
        return False
    if normalize_match(name) in _NOT_NAME or _AFFILIATION.search(name):
        return False
    if name.isupper():
        return False
    if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", name):
        return True
    tokens = name.split()
    if not 2 <= len(tokens) <= 5:
        return False
    capitals = 0
    for token in tokens:
        folded = token.casefold().strip(".")
        if folded in _STOP or folded in _NOT_NAME:
            return False
        if folded in _PARTICLE:
            continue
        if token.endswith("-") or token.endswith("—"):
            return False
        if not re.fullmatch(r"[A-Z][\w.'’-]*\.?", token):
            return False
        capitals += 1
    letters = [re.sub(r"[^A-Za-z]", "", token) for token in tokens]
    return capitals >= 2 and any(len(item) >= 2 for item in letters)


def _same(left: str, right: str) -> bool:
    return normalize_match(left) == normalize_match(right)


def _dedupe(names: list[str]) -> list[str]:
    chosen: list[str] = []
    seen: set[str] = set()
    for name in names:
        marker = normalize_match(name)
        if not marker or marker in seen:
            continue
        seen.add(marker)
        chosen.append(name)
    return chosen
