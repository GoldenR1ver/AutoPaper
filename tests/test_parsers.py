from autopaper.llm.prompts import coerce_array, parse_json_payload
from autopaper.sources.arxiv import parse_atom
from autopaper.sources.crossref import parse_crossref_items
from autopaper.sources.openalex import parse_openalex_author, parse_openalex_work
from autopaper.sources.pubmed import parse_pubmed_xml
from autopaper.sources.semantic_scholar import parse_s2_paper
from autopaper.textutil import normalize_doi, normalize_title, reconstruct_inverted_index


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <title>Graph  Neural
    Networks</title>
    <summary>A graph abstract.</summary>
    <published>2024-01-02T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/2401.00001v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.00001v2" title="pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
    <arxiv:doi>10.1000/ABC</arxiv:doi>
  </entry>
</feed>
"""

PUBMED = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>321</PMID>
      <Article>
        <Journal>
          <Title>Nature Methods</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Cells and graphs</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background text.</AbstractText>
          <AbstractText Label="RESULTS">Result text.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><ForeName>Grace</ForeName><LastName>Hopper</LastName></Author>
        </AuthorList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Neural Networks, Computer</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.2000/PMID</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_doi_title_and_inverted_index():
    assert normalize_doi("https://doi.org/10.1000/AbC") == "10.1000/abc"
    assert normalize_title("Time-Series Graph!") == "time series graph"
    assert reconstruct_inverted_index({"networks": [1], "Graph": [0]}) == "Graph networks"


def test_parse_arxiv_and_pubmed():
    paper = parse_atom(ATOM)[0]
    assert paper.title == "Graph Neural Networks"
    assert paper.arxiv_id == "2401.00001"
    assert paper.doi == "10.1000/abc"
    assert paper.year == 2024
    assert paper.pdf_url.endswith("2401.00001v2")
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert "cs.LG" in paper.categories

    pubmed = parse_pubmed_xml(PUBMED)[0]
    assert pubmed.pubmed_id == "321"
    assert pubmed.doi == "10.2000/pmid"
    assert "BACKGROUND" in pubmed.abstract
    assert pubmed.authors == ["Grace Hopper"]
    assert pubmed.categories == ["Neural Networks, Computer"]


def test_parse_crossref_openalex_and_s2():
    crossref = parse_crossref_items(
        [
            {
                "DOI": "10.3000/X",
                "title": ["A &amp; B"],
                "author": [{"given": "Grace", "family": "Hopper"}],
                "abstract": "<jats:p>Hello&nbsp;world</jats:p>",
                "issued": {"date-parts": [[2020]]},
                "container-title": ["JMLR"],
                "URL": "https://example.org/paper",
                "is-referenced-by-count": 4,
                "subject": ["Machine learning"],
            }
        ]
    )[0]
    assert crossref.doi == "10.3000/x"
    assert crossref.year == 2020
    assert crossref.venue == "JMLR"
    assert "Hello" in crossref.abstract
    assert crossref.citation_count == 4

    work = parse_openalex_work(
        {
            "id": "https://openalex.org/W9",
            "display_name": "Open work",
            "publication_year": 2022,
            "cited_by_count": 8,
            "doi": "https://doi.org/10.9/open",
            "abstract_inverted_index": {"Open": [0], "abstract": [1]},
            "authorships": [{"author": {"display_name": "Yann LeCun"}}],
            "primary_location": {
                "landing_page_url": "https://example.org/oa",
                "pdf_url": "https://example.org/oa.pdf",
                "source": {"display_name": "NeurIPS"},
            },
            "topics": [{"display_name": "Graph neural network"}],
            "ids": {"pmid": "999"},
        }
    )
    assert work.openalex_id == "W9"
    assert work.abstract == "Open abstract"
    assert work.venue == "NeurIPS"
    assert work.pubmed_id == "999"
    assert work.authors == ["Yann LeCun"]

    author = parse_openalex_author(
        {
            "id": "https://openalex.org/A12",
            "display_name": "Yann LeCun",
            "works_count": 10,
            "cited_by_count": 20,
            "summary_stats": {"h_index": 3},
            "last_known_institutions": [{"display_name": "NYU"}],
            "topics": [{"id": "https://openalex.org/T7", "display_name": "Deep learning"}],
        }
    )
    assert author.openalex_id == "A12"
    assert author.topic_ids == ["T7"]
    assert author.affiliation == "NYU"

    scholar = parse_s2_paper(
        {
            "paperId": "abc",
            "title": "Scholar paper",
            "abstract": "About graphs",
            "year": 2019,
            "venue": "ICLR",
            "citationCount": 2,
            "url": "https://www.semanticscholar.org/paper/abc",
            "externalIds": {"DOI": "10.8/S2", "ArXiv": "1901.00001"},
            "authors": [{"name": "Geoff Hinton"}],
            "openAccessPdf": {"url": "https://example.org/s2.pdf"},
            "fieldsOfStudy": ["Computer Science"],
        }
    )
    assert scholar.s2_id == "abc"
    assert scholar.doi == "10.8/s2"
    assert scholar.arxiv_id == "1901.00001"
    assert scholar.pdf_url.endswith("s2.pdf")


def test_parse_json_payload_accepts_fences_and_wrapped_arrays():
    assert parse_json_payload('说明\n```json\n{"a": 1}\n```') == {"a": 1}
    assert coerce_array({"papers": [{"key": "k"}]}) == [{"key": "k"}]
