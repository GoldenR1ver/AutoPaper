from pathlib import Path

import pytest

from autopaper.config import load_settings
from autopaper.errors import ConfigError, LLMError
from autopaper.models import Paper
from autopaper.pipeline.collect import run_collect


def _clear_embedding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AUTOPAPER_EMBEDDING_BASE_URL",
        "AUTOPAPER_EMBEDDING_API_KEY",
        "AUTOPAPER_EMBEDDING_MODEL",
        "AUTOPAPER_DISABLE_EMBEDDING",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_env_and_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "autopaper.yaml").write_text(
        "data_dir: from-yaml\nmailto: you@example.com\nllm:\n  model: from-yaml\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("AUTOPAPER_CONFIG", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AUTOPAPER_DATA_DIR", str(tmp_path / "from-env"))
    monkeypatch.setenv("AUTOPAPER_MODEL", "from-env")
    settings = load_settings()
    assert settings.data_dir == (tmp_path / "from-env").resolve()
    assert settings.mailto == ""
    assert settings.llm_model == "from-env"
    assert settings.llm_api_key == "from-dotenv"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_base_url == settings.llm_base_url
    assert settings.recall_similarity == 0.42

    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert load_settings().llm_api_key == "from-env"

    (tmp_path / "bad.yaml").write_text(":\n  -", encoding="utf-8")
    monkeypatch.setenv("AUTOPAPER_CONFIG", str(tmp_path / "bad.yaml"))
    with pytest.raises(ConfigError):
        load_settings()


def test_deepseek_key_selects_v41_flash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("AUTOPAPER_MODEL", raising=False)
    monkeypatch.delenv("AUTOPAPER_CONFIG", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("AUTOPAPER_DATA_DIR", str(tmp_path / "data"))
    settings = load_settings()
    assert settings.llm_api_key == "sk-test"
    assert settings.llm_model == "deepseek-flash"
    assert settings.llm_base_url == "https://api.deepseek.com"
    assert settings.llm_provider == "deepseek"
    assert settings.embedding_base_url == ""
    assert settings.embedding_model == ""

    monkeypatch.setenv("OPENAI_API_KEY", "")
    blocked = load_settings()
    assert blocked.llm_api_key == ""
    assert blocked.llm_model == "gpt-4o-mini"


def test_collect_pipeline_dedups_filters_and_exports(tmp_path: Path):
    from autopaper.models import Topic
    from autopaper.storage.db import KnowledgeBase

    class FakeSource:
        name = "arxiv"

        def search(self, query, *, limit, since_year=None, sort="relevance"):
            assert query == "graph neural networks"
            return [
                Paper(
                    title="Graph Neural Networks for Time Series",
                    abstract="graph neural networks for forecasting",
                    year=2024,
                    doi="10.1000/aaa",
                    authors=["Ada Lovelace"],
                    source="arxiv",
                ),
                Paper(
                    title="Graph Neural Networks for Time Series",
                    abstract="graph neural networks with a longer abstract",
                    doi="10.1000/AAA",
                    citation_count=4,
                    source="crossref",
                ),
                Paper(title="A survey of unrelated pasta", abstract="boiling", year=2020, source="pubmed"),
            ]

    class Boom:
        def complete_json(self, *args, **kwargs):
            raise LLMError("down")

    topic = Topic(
        id="gnn",
        name="GNN",
        query="graph neural networks",
        keywords=["graph neural networks"],
        exclude_keywords=["pasta"],
        sources=["arxiv"],
        llm_top_n=5,
    )
    settings = type("S", (), {})()
    settings.data_dir = tmp_path
    settings.similarity_threshold = 92
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        kb.save_topic(topic)
        stats = run_collect(topic, kb, settings, sources=[FakeSource()], llm=Boom(), export=True)
        assert stats.raw_count == 3
        assert stats.deduped_count == 2
        assert stats.filtered_count == 1
        assert stats.stored_count == 1
        assert stats.candidate_count == 1
        assert stats.llm_scored == 0
        assert stats.status == "partial"
        assert any("模型打分失败" in item for item in stats.errors)
        exported = Path(stats.export_dir)
        assert (exported / "papers.csv").exists()
        assert (exported / "papers.bib").exists()
        assert (exported / "candidates.md").exists()
        assert kb.get_topic("GNN").last_run_at is not None
        assert stats.recalled_count == 0


def test_collect_recalls_a_synonym_the_keywords_missed(tmp_path: Path):
    from autopaper.models import Topic
    from autopaper.storage.db import KnowledgeBase

    class FakeSource:
        name = "arxiv"

        def search(self, query, *, limit, since_year=None, sort="relevance"):
            return [
                Paper(title="Scientometrics of software", abstract="scientometrics of packages", year=2024, source="arxiv"),
                Paper(
                    title="Science of science and semantics",
                    abstract="how scientific fields cite each other",
                    year=2024,
                    source="arxiv",
                ),
                Paper(title="Science of cooking", abstract="pasta boiling", year=2020, source="arxiv"),
            ]

    class EmbedLLM:
        def __init__(self):
            self.user = ""

        def embed(self, texts):
            vectors = []
            for index, text in enumerate(texts):
                if index == 0 or "cite each other" in text.casefold():
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

        def complete_json(self, system, user, *, expect="object", max_tokens=2048):
            self.user = user
            import re

            keys = re.findall(r"^\[(.+)\]$", user, re.MULTILINE)
            return [{"key": key, "score": 70, "reason": "语义相关", "keep": True} for key in keys]

    topic = Topic(
        id="sci",
        name="scientometrics",
        query="scientometrics",
        keywords=["scientometrics"],
        exclude_keywords=["pasta"],
        sources=["arxiv"],
        llm_top_n=1,
    )
    settings = type("S", (), {})()
    settings.data_dir = tmp_path
    settings.similarity_threshold = 92
    llm = EmbedLLM()
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        kb.save_topic(topic)
        stats = run_collect(topic, kb, settings, sources=[FakeSource()], llm=llm, export=False)
    assert stats.recalled_count == 1
    assert stats.filtered_count == 2
    assert stats.stored_count == 2
    assert stats.llm_scored == 1
    assert "Science of science" in llm.user
    assert "pasta" not in llm.user.casefold()


def test_embedding_client_restores_response_order(tmp_path: Path):
    import json

    import httpx

    from autopaper.config import Settings
    from autopaper.llm.client import EmbeddingClient

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["a", "b"]
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]},
        )

    configured = Settings(
        data_dir=tmp_path,
        embedding_api_key="k",
        embedding_base_url="https://example.test/v1",
        embedding_model="text-embedding-3-small",
    )
    client = EmbeddingClient(configured, transport=httpx.MockTransport(handler))
    try:
        assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    finally:
        client.close()
