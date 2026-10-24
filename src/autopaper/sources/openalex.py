from __future__ import annotations

from autopaper.http import HttpClient
from autopaper.models import Paper, ResearcherProfile
from autopaper.textutil import (
    collapse_ws,
    dedupe_names,
    first_affiliation,
    normalize_orcid,
    openalex_short_id,
    reconstruct_inverted_index,
)

WORKS_URL = "https://api.openalex.org/works"
AUTHORS_URL = "https://api.openalex.org/authors"
WORK_SELECT = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "cited_by_count",
        "abstract_inverted_index",
        "authorships",
        "primary_location",
        "topics",
        "concepts",
        "ids",
        "open_access",
    ]
)


def _categories(item: dict) -> list[str]:
    names: list[str] = []
    for bucket in ("topics", "concepts"):
        for entry in item.get(bucket) or []:
            if isinstance(entry, dict) and entry.get("display_name"):
                names.append(entry["display_name"])
    return dedupe_names(names)


def parse_openalex_work(item: dict) -> Paper | None:
    title = collapse_ws(item.get("display_name") or item.get("title"))
    if not title:
        return None
    authors: list[str] = []
    for authorship in item.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author") or {}
        name = collapse_ws(author.get("display_name"))
        if name:
            authors.append(name)
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    open_access = item.get("open_access") or {}
    ids = item.get("ids") or {}
    abstract = item.get("abstract") or reconstruct_inverted_index(item.get("abstract_inverted_index"))
    return Paper(
        title=title,
        abstract=abstract or "",
        authors=authors,
        year=item.get("publication_year"),
        doi=item.get("doi") or ids.get("doi"),
        pubmed_id=ids.get("pmid"),
        openalex_id=item.get("id"),
        url=location.get("landing_page_url") or open_access.get("oa_url"),
        pdf_url=location.get("pdf_url") or open_access.get("oa_url"),
        venue=source.get("display_name"),
        citation_count=item.get("cited_by_count"),
        categories=_categories(item),
        source="openalex",
        sources=["openalex"],
    ).cleanup()


def parse_openalex_author(item: dict) -> ResearcherProfile:
    concepts: list[str] = []
    topic_ids: list[str] = []
    for topic in item.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        if topic.get("display_name"):
            concepts.append(topic["display_name"])
        short = openalex_short_id(topic.get("id"))
        if short:
            topic_ids.append(short)
    for concept in item.get("x_concepts") or []:
        if isinstance(concept, dict) and concept.get("display_name"):
            concepts.append(concept["display_name"])
    stats = item.get("summary_stats") or {}
    return ResearcherProfile(
        name=collapse_ws(item.get("display_name") or "") or "未知作者",
        orcid=normalize_orcid(item.get("orcid")),
        affiliation=first_affiliation(item.get("last_known_institutions")),
        openalex_id=openalex_short_id(item.get("id")),
        h_index=stats.get("h_index"),
        cited_by_count=item.get("cited_by_count"),
        works_count=item.get("works_count"),
        concepts=dedupe_names(concepts)[:12],
        topic_ids=topic_ids[:5],
    )


class OpenAlexSource:
    name = "openalex"

    def __init__(self, http: HttpClient, mailto: str = ""):
        self.http = http
        self.mailto = mailto

    def _params(self, extra: dict) -> dict:
        params = dict(extra)
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def search(
        self,
        query: str,
        *,
        limit: int,
        since_year: int | None = None,
        sort: str = "relevance",
    ) -> list[Paper]:
        filters = ["is_paratext:false"]
        if since_year:
            filters.append(f"from_publication_date:{since_year}-01-01")
        papers: list[Paper] = []
        page = 1
        while len(papers) < limit:
            params = self._params(
                {
                    "search": query,
                    "filter": ",".join(filters),
                    "per-page": min(50, limit - len(papers)),
                    "page": page,
                    "select": WORK_SELECT,
                }
            )
            if sort == "date":
                params["sort"] = "publication_date:desc"
            data = self.http.get_json(WORKS_URL, params=params, bucket=self.name)
            results = data.get("results") or []
            if not results:
                break
            for item in results:
                paper = parse_openalex_work(item)
                if paper is not None:
                    papers.append(paper)
                if len(papers) >= limit:
                    break
            page += 1
            if len(results) < params["per-page"]:
                break
        return papers[:limit]

    def search_authors(self, name: str, *, limit: int = 5) -> list[ResearcherProfile]:
        params = self._params({"search": name, "per-page": limit})
        data = self.http.get_json(AUTHORS_URL, params=params, bucket=self.name)
        return [parse_openalex_author(item) for item in data.get("results") or []]

    def get_author(self, openalex_id: str) -> ResearcherProfile:
        short = openalex_short_id(openalex_id) or openalex_id
        data = self.http.get_json(
            f"{AUTHORS_URL}/{short}",
            params=self._params({}),
            bucket=self.name,
        )
        return parse_openalex_author(data)

    def author_works(self, openalex_id: str, *, limit: int, sort: str) -> list[Paper]:
        short = openalex_short_id(openalex_id) or openalex_id
        params = self._params(
            {
                "filter": f"authorships.author.id:{short},is_paratext:false",
                "sort": sort,
                "per-page": min(limit, 50),
                "select": WORK_SELECT,
            }
        )
        data = self.http.get_json(WORKS_URL, params=params, bucket=self.name)
        papers: list[Paper] = []
        for item in data.get("results") or []:
            paper = parse_openalex_work(item)
            if paper is not None:
                papers.append(paper)
        return papers

    def related_authors(self, topic_id: str, *, limit: int = 10) -> list[ResearcherProfile]:
        short = openalex_short_id(topic_id) or topic_id
        params = self._params(
            {
                "filter": f"topics.id:{short}",
                "sort": "cited_by_count:desc",
                "per-page": limit,
            }
        )
        data = self.http.get_json(AUTHORS_URL, params=params, bucket=self.name)
        return [parse_openalex_author(item) for item in data.get("results") or []]
