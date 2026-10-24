from __future__ import annotations

import xml.etree.ElementTree as ET

from autopaper.errors import SourceError
from autopaper.http import HttpClient
from autopaper.models import Paper
from autopaper.sources.common import child_text, direct_child, direct_children
from autopaper.textutil import dedupe_names

API_URL = "https://export.arxiv.org/api/query"


def build_arxiv_query(query: str, since_year: int | None) -> str:
    text = query.strip()
    if not text:
        raise SourceError("arxiv", "检索式为空")
    has_field = any(token in text for token in ("all:", "ti:", "au:", "abs:", "cat:"))
    if has_field:
        base = text
    else:
        phrase = text.replace('"', "")
        base = f'all:"{phrase}"'
    if since_year:
        base = f"({base}) AND submittedDate:[{since_year}01010000 TO 209912312359]"
    return base


def parse_atom(xml_text: str) -> list[Paper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SourceError("arxiv", f"无法解析 Atom：{exc}") from exc
    papers: list[Paper] = []
    for entry in direct_children(root, "entry"):
        paper = _parse_entry(entry)
        if paper is not None:
            papers.append(paper.cleanup())
    return papers


def _parse_entry(entry) -> Paper | None:
    title = child_text(entry, "title")
    if not title or title.casefold() == "error":
        return None
    url = None
    pdf_url = None
    for link in direct_children(entry, "link"):
        href = link.attrib.get("href")
        if not href:
            continue
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = href
        elif link.attrib.get("rel") == "alternate":
            url = href
    identifier = child_text(entry, "id")
    url = url or identifier or None
    authors = []
    for author in direct_children(entry, "author"):
        name = child_text(author, "name")
        if name:
            authors.append(name)
    categories: list[str] = []
    primary = direct_child(entry, "primary_category")
    if primary is not None and primary.attrib.get("term"):
        categories.append(primary.attrib["term"])
    for category in direct_children(entry, "category"):
        term = category.attrib.get("term")
        if term:
            categories.append(term)
    published = child_text(entry, "published")
    year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
    return Paper(
        title=title,
        abstract=child_text(entry, "summary"),
        authors=authors,
        year=year,
        doi=child_text(entry, "doi") or None,
        arxiv_id=identifier or url,
        url=url,
        pdf_url=pdf_url,
        categories=dedupe_names(categories),
        source="arxiv",
        sources=["arxiv"],
    )


class ArxivSource:
    name = "arxiv"

    def __init__(self, http: HttpClient):
        self.http = http

    def search(
        self,
        query: str,
        *,
        limit: int,
        since_year: int | None = None,
        sort: str = "relevance",
    ) -> list[Paper]:
        search_query = build_arxiv_query(query, since_year)
        papers: list[Paper] = []
        start = 0
        while len(papers) < limit:
            batch = min(50, limit - len(papers))
            params = {
                "search_query": search_query,
                "start": start,
                "max_results": batch,
                "sortBy": "submittedDate" if sort == "date" else "relevance",
                "sortOrder": "descending",
            }
            text = self.http.get_text(API_URL, params=params, bucket=self.name)
            found = parse_atom(text)
            if not found:
                break
            papers.extend(found)
            start += len(found)
            if len(found) < batch:
                break
        return papers[:limit]
