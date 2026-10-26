from __future__ import annotations

import re
from dataclasses import dataclass

from autopaper.analysis.frontmatter import parse_authors, parse_keywords
from autopaper.textutil import collapse_ws

AUTHOR_LIMIT = 800
SECTION_LIMIT = 1200

_KINDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("abstract", ("abstract", "summary", "摘要")),
    ("intro", ("introduction", "background", "引言")),
    ("keywords", ("keywords", "keyword", "index terms", "关键词")),
    ("conclusion", ("conclusion", "conclusions", "concluding remarks", "结论", "总结")),
    ("end", ("references", "bibliography", "acknowledgements", "acknowledgments", "参考文献", "致谢", "appendix", "附录")),
)


@dataclass(frozen=True)
class PaperExcerpt:
    name: str
    authors_text: str
    abstract: str
    conclusion: str
    front: str
    author_names: tuple[str, ...] = ()
    corresponding: str = ""
    keywords: tuple[str, ...] = ()

    def author_haystack(self) -> str:
        return "\n".join(part for part in (self.name, self.authors_text) if part)

    def topic_haystack(self) -> str:
        parts = [self.name, self.abstract, self.conclusion]
        if not self.abstract and not self.conclusion:
            parts.append(self.front)
        return "\n".join(part for part in parts if part)

    def llm_text(self) -> str:
        lines = [
            f"文件名：{self.name or '未命名'}",
            f"作者栏：{self.authors_text or '未提取到'}",
            f"摘要：{self.abstract or '未提取到'}",
            f"结论：{self.conclusion or '未提取到'}",
        ]
        if not self.abstract and not self.conclusion and self.front:
            lines.append(f"文首：{self.front}")
        return "\n".join(lines)


def build_excerpt(name: str, text: str, metadata_author: str = "") -> PaperExcerpt:
    body = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    headings = _headings(body)
    abstract_heading = _first(headings, "abstract", 0)
    conclusion_after = abstract_heading[0] if abstract_heading else 0
    conclusion_heading = _first(headings, "conclusion", conclusion_after)
    abstract = _body(body, headings, abstract_heading)
    conclusion = _body(body, headings, conclusion_heading)
    front_source = body[: abstract_heading[0]] if abstract_heading else body
    front = _clip_head(front_source, AUTHOR_LIMIT)
    author_bits = [collapse_ws(metadata_author), front]
    authors_text = _clip_head(" ".join(bit for bit in author_bits if bit), AUTHOR_LIMIT + 200)
    keyword_heading = _first(headings, "keywords", 0)
    author_names, corresponding = parse_authors(front_source, body, metadata_author)
    keywords = parse_keywords(_raw_section(body, headings, keyword_heading))
    return PaperExcerpt(
        name=name,
        authors_text=authors_text,
        abstract=abstract,
        conclusion=conclusion,
        front=front,
        author_names=author_names,
        corresponding=corresponding,
        keywords=keywords,
    )


def _headings(text: str) -> list[tuple[int, str, int]]:
    found: list[tuple[int, str, int]] = []
    for kind, words in _KINDS:
        pattern = re.compile(
            r"(?im)(?:^|\n)\s*(?:\d{1,2}(?:\.\d+){0,3}[.)]\s+)?(?:"
            + "|".join(re.escape(word) for word in words)
            + r")\b\s*[:.\-—–]?\s*"
        )
        for match in pattern.finditer(text):
            found.append((match.start(), kind, match.end()))
    found.sort()
    return found


def _first(headings: list[tuple[int, str, int]], kind: str, after: int) -> tuple[int, str, int] | None:
    for item in headings:
        if item[1] == kind and item[0] >= after:
            return item
    return None


def _raw_section(text: str, headings: list[tuple[int, str, int]], heading: tuple[int, str, int] | None) -> str:
    if heading is None:
        return ""
    later = [item[0] for item in headings if item[0] > heading[0]]
    stop = later[0] if later else min(len(text), heading[2] + 500)
    kept: list[str] = []
    for line in text[heading[2] : stop].splitlines():
        if re.match(r"\s*(?:I{1,3}\.|[1-9]\d?\.\s|ACM Reference|ABSTRACT|Introduction|参考文献)", line):
            break
        if line.strip():
            kept.append(line.strip())
        if len(kept) >= 4:
            break
    return "\n".join(kept)


def _body(text: str, headings: list[tuple[int, str, int]], heading: tuple[int, str, int] | None) -> str:
    if heading is None:
        return ""
    later = [item[0] for item in headings if item[0] > heading[0]]
    stop = later[0] if later else heading[2] + SECTION_LIMIT
    return _clip_head(text[heading[2] : stop], SECTION_LIMIT)


def _clip_head(text: str, limit: int) -> str:
    body = collapse_ws(text)
    if len(body) <= limit:
        return body
    return body[:limit].rstrip()
