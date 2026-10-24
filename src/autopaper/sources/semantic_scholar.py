from __future__ import annotations

from urllib.parse import quote

from autopaper.errors import SourceError
from autopaper.http import HttpClient
from autopaper.models import Paper, ResearcherProfile
from autopaper.textutil import collapse_ws, dedupe_names, first_affiliation

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
AUTHOR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/author/search"
PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/"
AUTHOR_URL = "https://api.semanticscholar.org/graph/v1/author/"
PAPER_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "year",
        "venue",
        "citationCount",
        "url",
        "externalIds",
        "authors",
        "openAccessPdf",
        "fieldsOfStudy",
    ]
)


def _names(value) -> list[str]:
    names: list[str] = []
    for item in value or []:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("category") or item.get("name") or item.get("display_name")
            if text:
                names.append(str(text))
    return dedupe_names(names)


def _author_names(value) -> list[str]:
    names: list[str] = []
    for item in value or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("name"):
            names.append(item["name"])
    return dedupe_names(names)


def parse_s2_paper(entry: dict | None) -> Paper | None:
    if not isinstance(entry, dict):
        return None
    title = collapse_ws(entry.get("title"))
    if not title:
        return None
    external = entry.get("externalIds") or {}
    pdf = entry.get("openAccessPdf") or {}
    citations = entry.get("citationCount")
    return Paper(
        title=title,
        abstract=entry.get("abstract") or "",
        authors=_author_names(entry.get("authors")),
        year=entry.get("year") if isinstance(entry.get("year"), int) else None,
        doi=external.get("DOI"),
        arxiv_id=external.get("ArXiv"),
        pubmed_id=external.get("PubMed"),
        s2_id=entry.get("paperId"),
        url=entry.get("url"),
        pdf_url=pdf.get("url") if isinstance(pdf, dict) else None,
        venue=collapse_ws(entry.get("venue")) or None,
        citation_count=citations if isinstance(citations, int) else None,
        categories=_names(entry.get("fieldsOfStudy")),
        source="semantic_scholar",
        sources=["semantic_scholar"],
    ).cleanup()


def parse_s2_author(entry: dict) -> ResearcherProfile:
    return ResearcherProfile(
        name=collapse_ws(entry.get("name") or "") or "未知作者",
        affiliation=first_affiliation(entry.get("affiliations")),
        s2_id=entry.get("authorId"),
        h_index=entry.get("hIndex"),
        cited_by_count=entry.get("citationCount"),
        works_count=entry.get("paperCount"),
    )


def s2_lookup_id(paper: Paper) -> str | None:
    if paper.s2_id:
        return paper.s2_id
    if paper.doi:
        return f"DOI:{paper.doi}"
    if paper.arxiv_id:
        return f"ARXIV:{paper.arxiv_id}"
    if paper.pubmed_id:
        return f"PMID:{paper.pubmed_id}"
    return None


class SemanticScholarSource:
    name = "semantic_scholar"

    def __init__(self, http: HttpClient, api_key: str = ""):
        self.http = http
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"x-api-key": self.api_key}
        return {}

    def _data(self, url: str, params: dict) -> dict:
        data = self.http.get_json(url, params=params, headers=self._headers(), bucket=self.name)
        if isinstance(data, dict) and "message" in data and "data" not in data and "paperId" not in data:
            raise SourceError(self.name, str(data.get("message")))
        if isinstance(data, dict) and data.get("error") and "data" not in data:
            raise SourceError(self.name, str(data.get("error")))
        return data

    def search(
        self,
        query: str,
        *,
        limit: int,
        since_year: int | None = None,
        sort: str = "relevance",
    ) -> list[Paper]:
        papers: list[Paper] = []
        offset = 0
        while len(papers) < limit:
            batch = min(100, limit - len(papers))
            params: dict = {
                "query": query,
                "limit": batch,
                "offset": offset,
                "fields": PAPER_FIELDS,
            }
            if since_year:
                params["year"] = f"{since_year}-"
            data = self._data(SEARCH_URL, params)
            rows = data.get("data") or []
            if not rows:
                break
            for row in rows:
                paper = parse_s2_paper(row)
                if paper is not None:
                    papers.append(paper)
            offset += len(rows)
            if len(rows) < batch:
                break
        return papers[:limit]

    def search_authors(self, name: str, *, limit: int = 5) -> list[ResearcherProfile]:
        params = {
            "query": name,
            "limit": limit,
            "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex",
        }
        data = self._data(AUTHOR_SEARCH_URL, params)
        return [parse_s2_author(item) for item in data.get("data") or []]

    def author_papers(self, author_id: str, *, limit: int = 50) -> list[Paper]:
        params = {"limit": min(limit, 100), "fields": PAPER_FIELDS}
        url = AUTHOR_URL + quote(author_id, safe="") + "/papers"
        data = self._data(url, params)
        papers: list[Paper] = []
        for row in data.get("data") or []:
            paper = parse_s2_paper(row)
            if paper is not None:
                papers.append(paper)
        return papers

    def references(self, paper_id: str, *, limit: int = 30) -> list[Paper]:
        fields = ",".join(
            [
                "references.paperId",
                "references.title",
                "references.year",
                "references.authors",
                "references.externalIds",
                "references.url",
                "references.venue",
                "references.citationCount",
            ]
        )
        url = PAPER_URL + quote(paper_id, safe="")
        data = self._data(url, {"fields": fields})
        papers: list[Paper] = []
        for row in data.get("references") or []:
            paper = parse_s2_paper(row)
            if paper is not None:
                papers.append(paper)
            if len(papers) >= limit:
                break
        return papers
