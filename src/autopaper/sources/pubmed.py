from __future__ import annotations

import xml.etree.ElementTree as ET

from autopaper.errors import SourceError
from autopaper.http import HttpClient
from autopaper.models import Paper
from autopaper.sources.common import child_text, direct_child, direct_children, local_name
from autopaper.textutil import collapse_ws

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_pubmed_xml(xml_text: str) -> list[Paper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SourceError("pubmed", f"无法解析 PubMed XML：{exc}") from exc
    papers: list[Paper] = []
    for article in root.iter():
        if local_name(article.tag) != "PubmedArticle":
            continue
        paper = _parse_article(article)
        if paper is not None:
            papers.append(paper.cleanup())
    return papers


def _parse_article(article) -> Paper | None:
    medline = direct_child(article, "MedlineCitation")
    pubmed_data = direct_child(article, "PubmedData")
    art = direct_child(medline, "Article")
    if art is None:
        return None
    title = child_text(art, "ArticleTitle")
    if not title:
        return None
    journal = direct_child(art, "Journal")
    venue = child_text(journal, "Title") or None
    issue = direct_child(journal, "JournalIssue")
    pub_date = direct_child(issue, "PubDate")
    year_text = child_text(pub_date, "Year") or child_text(pub_date, "MedlineDate")
    year = int(year_text[:4]) if len(year_text) >= 4 and year_text[:4].isdigit() else None
    abstract_node = direct_child(art, "Abstract")
    abstract_parts = []
    for part in direct_children(abstract_node, "AbstractText"):
        label = part.attrib.get("Label")
        text = collapse_ws("".join(part.itertext()))
        if text:
            abstract_parts.append(f"{label}: {text}" if label else text)
    authors = []
    author_list = direct_child(art, "AuthorList")
    for author in direct_children(author_list, "Author"):
        collective = child_text(author, "CollectiveName")
        if collective:
            authors.append(collective)
            continue
        name = collapse_ws(f"{child_text(author, 'ForeName')} {child_text(author, 'LastName')}")
        if name:
            authors.append(name)
    categories = []
    mesh = direct_child(medline, "MeshHeadingList")
    for heading in direct_children(mesh, "MeshHeading"):
        descriptor = child_text(heading, "DescriptorName")
        if descriptor:
            categories.append(descriptor)
    pmid = child_text(medline, "PMID")
    doi = _article_id(pubmed_data, "doi") or _elocation_doi(art)
    return Paper(
        title=title,
        abstract=" ".join(abstract_parts),
        authors=authors,
        year=year,
        doi=doi,
        pubmed_id=pmid,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
        venue=venue,
        categories=categories,
        source="pubmed",
        sources=["pubmed"],
    )


def _article_id(pubmed_data, id_type: str) -> str | None:
    article_ids = direct_child(pubmed_data, "ArticleIdList")
    for node in direct_children(article_ids, "ArticleId"):
        if node.attrib.get("IdType") == id_type:
            text = collapse_ws("".join(node.itertext()))
            if text:
                return text
    return None


def _elocation_doi(article) -> str | None:
    for node in direct_children(article, "ELocationID"):
        if node.attrib.get("EIdType") == "doi":
            text = collapse_ws("".join(node.itertext()))
            if text:
                return text
    return None


class PubMedSource:
    name = "pubmed"

    def __init__(self, http: HttpClient, email: str = "", api_key: str = ""):
        self.http = http
        self.email = email
        self.api_key = api_key

    def _common(self) -> dict:
        params = {"tool": "autopaper", "db": "pubmed"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search(
        self,
        query: str,
        *,
        limit: int,
        since_year: int | None = None,
        sort: str = "relevance",
    ) -> list[Paper]:
        term = query.strip()
        if since_year:
            term = f"({term}) AND {since_year}:3000[dp]"
        params = self._common()
        params.update(
            {
                "term": term,
                "retmax": limit,
                "retmode": "json",
                "sort": "pub_date" if sort == "date" else "relevance",
            }
        )
        data = self.http.get_json(ESEARCH, params=params, bucket=self.name)
        result = data.get("esearchresult") or {}
        if result.get("ERROR"):
            raise SourceError(self.name, str(result["ERROR"]))
        ids = result.get("idlist") or []
        papers: list[Paper] = []
        for start in range(0, len(ids), 80):
            chunk = ids[start : start + 80]
            fetch_params = self._common()
            fetch_params.update({"id": ",".join(chunk), "retmode": "xml"})
            xml_text = self.http.get_text(EFETCH, params=fetch_params, bucket=self.name)
            papers.extend(parse_pubmed_xml(xml_text))
        return papers[:limit]
