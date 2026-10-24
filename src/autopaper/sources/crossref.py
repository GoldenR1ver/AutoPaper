from __future__ import annotations

from autopaper.http import HttpClient
from autopaper.models import Paper
from autopaper.textutil import collapse_ws, strip_markup

API_URL = "https://api.crossref.org/works"
SELECT = ",".join(
    [
        "DOI",
        "title",
        "author",
        "abstract",
        "issued",
        "published-print",
        "published-online",
        "container-title",
        "URL",
        "is-referenced-by-count",
        "link",
        "subject",
        "type",
    ]
)


def _year(entry: dict) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = ((entry.get(key) or {}).get("date-parts") or [[]])[0]
        if not parts or parts[0] in (None, ""):
            continue
        try:
            return int(parts[0])
        except (TypeError, ValueError):
            continue
    return None


def _author_name(entry: dict) -> str | None:
    if not isinstance(entry, dict):
        return None
    if entry.get("name"):
        return collapse_ws(entry["name"]) or None
    given = entry.get("given") or ""
    family = entry.get("family") or ""
    return collapse_ws(f"{given} {family}") or None


def parse_crossref_items(items: list[dict]) -> list[Paper]:
    papers: list[Paper] = []
    for item in items:
        titles = item.get("title") or []
        raw_title = " ".join(titles) if isinstance(titles, list) else str(titles)
        title = strip_markup(raw_title)
        if not title:
            continue
        authors = []
        for author in item.get("author") or []:
            name = _author_name(author)
            if name:
                authors.append(name)
        venues = item.get("container-title") or []
        venue = collapse_ws(venues[0]) if venues else None
        pdf_url = None
        for link in item.get("link") or []:
            if not isinstance(link, dict):
                continue
            if link.get("content-type") == "application/pdf" and link.get("URL"):
                pdf_url = link["URL"]
                break
        subjects = [collapse_ws(subject) for subject in item.get("subject") or [] if collapse_ws(str(subject))]
        citations = item.get("is-referenced-by-count")
        papers.append(
            Paper(
                title=title,
                abstract=strip_markup(item.get("abstract")),
                authors=authors,
                year=_year(item),
                doi=item.get("DOI"),
                url=item.get("URL"),
                pdf_url=pdf_url,
                venue=venue,
                citation_count=int(citations) if isinstance(citations, int) else None,
                categories=subjects,
                source="crossref",
                sources=["crossref"],
            ).cleanup()
        )
    return papers


class CrossrefSource:
    name = "crossref"

    def __init__(self, http: HttpClient, mailto: str = ""):
        self.http = http
        self.mailto = mailto

    def search(
        self,
        query: str,
        *,
        limit: int,
        since_year: int | None = None,
        sort: str = "relevance",
    ) -> list[Paper]:
        items: list[dict] = []
        offset = 0
        while len(items) < limit:
            rows = min(50, limit - len(items))
            params: dict = {
                "query": query,
                "rows": rows,
                "offset": offset,
                "select": SELECT,
            }
            if self.mailto:
                params["mailto"] = self.mailto
            if since_year:
                params["filter"] = f"from-pub-date:{since_year}-01-01"
            if sort == "date":
                params["sort"] = "published"
                params["order"] = "desc"
            data = self.http.get_json(API_URL, params=params, bucket=self.name)
            batch = (data.get("message") or {}).get("items") or []
            if not batch:
                break
            items.extend(batch)
            offset += len(batch)
            if len(batch) < rows:
                break
        return parse_crossref_items(items)[:limit]
