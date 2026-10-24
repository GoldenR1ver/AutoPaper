import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autopaper.catalog import format_author, resolve_authors, resolve_topics
from autopaper.cli import app
from autopaper.errors import AutoPaperError
from autopaper.storage.db import SCHEMA, KnowledgeBase

runner = CliRunner()


def test_author_alias_display_and_conflict(tmp_path: Path):
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        author = kb.create_author("Huilian Sophie Qiu", ["Huilian Qiu", "Qiu Huilian"])
        author = kb.update_aliases(author.id, add=["邱惠莲"])
        assert format_author(author) == "Huilian Sophie Qiu（Huilian Qiu / Qiu Huilian / 邱惠莲）"
        assert kb.find_author("邱惠莲").id == author.id
        assert kb.find_author("huilian qiu").id == author.id
        other = kb.create_author("Other Person")
        with pytest.raises(AutoPaperError):
            kb.update_aliases(other.id, add=["邱惠莲"])
        removed = kb.update_aliases(author.id, remove=["Qiu Huilian"])
        assert format_author(removed) == "Huilian Sophie Qiu（Huilian Qiu / 邱惠莲）"
        chosen = resolve_authors(kb, "邱惠莲;Other Person")
        assert [item.name for item in chosen] == ["Huilian Sophie Qiu", "Other Person"]


def test_selection_uses_numbers_and_creates_topics(tmp_path: Path):
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        kb.create_author("Ada Lovelace")
        kb.create_author("Huilian Sophie Qiu")
        chosen = resolve_authors(kb, "2;1")
        assert [item.name for item in chosen] == ["Huilian Sophie Qiu", "Ada Lovelace"]
        topics = resolve_topics(kb, "time series;graph reading")
        again = resolve_topics(kb, "1;time series")
        assert [topic.name for topic in topics] == ["time series", "graph reading"]
        assert [topic.name for topic in again] == ["graph reading", "time series"]


def test_upload_groups_by_title_and_skips_duplicates(tmp_path: Path):
    inbox = tmp_path / "inbox"
    nested = inbox / "more"
    nested.mkdir(parents=True)
    (inbox / "Alpha Paper.txt").write_text("alpha", encoding="utf-8")
    (nested / "Alpha Paper.md").write_text("duplicate", encoding="utf-8")
    (inbox / "Beta Paper.txt").write_text("beta", encoding="utf-8")
    with KnowledgeBase(tmp_path / "kb.sqlite") as kb:
        from autopaper.library import plan_uploads, store_uploads

        planned = plan_uploads(inbox, kb, tmp_path)
        assert sum(not item.skip for item in planned) == 2
        assert sum(item.skip for item in planned) == 1
        authors = resolve_authors(kb, "Huilian Sophie Qiu")
        topics = resolve_topics(kb, "time series")
        stored = store_uploads(planned, kb, authors, topics)
        uploaded = [item for item in stored if not item.skip]
        assert (tmp_path / "library" / "alpha-paper" / "Alpha Paper.txt").exists()
        assert (tmp_path / "library" / "beta-paper" / "Beta Paper.txt").exists()
        assert "Alpha Paper" in (tmp_path / "library" / "alpha-paper" / "title.txt").read_text(encoding="utf-8")
        again = plan_uploads(inbox, kb, tmp_path)
        assert all(item.skip for item in again)
        assert kb.has_title(uploaded[0].title)


def test_upgrade_from_version_1_adds_author_tables(tmp_path: Path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO schema_version(version) VALUES (1)")
    conn.commit()
    conn.close()
    with KnowledgeBase(path) as kb:
        columns = [row[1] for row in kb.conn.execute("PRAGMA table_info(papers)")]
        assert "local_path" in columns
        kb.create_author("Ada Lovelace")
        assert kb.find_author("Ada Lovelace") is not None


def test_cli_upload_and_alias():
    with runner.isolated_filesystem():
        inbox = Path("inbox")
        inbox.mkdir()
        (inbox / "Alpha Paper.txt").write_text("alpha", encoding="utf-8")
        (inbox / "Beta Paper.txt").write_text("beta", encoding="utf-8")
        env = {
            "AUTOPAPER_CONFIG": str(Path("missing.yaml").resolve()),
            "AUTOPAPER_DATA_DIR": "data",
            "AUTOPAPER_DISABLE_LLM": "1",
            "OPENAI_API_KEY": "",
        }
        uploaded = runner.invoke(
            app,
            ["upload", "inbox", "--author", "Huilian Sophie Qiu", "--topic", "time series;graph reading"],
            env=env,
        )
        assert uploaded.exit_code == 0, uploaded.output
        assert "上传 2 篇，跳过 0 篇。" in uploaded.output
        assert "Huilian Sophie Qiu" in uploaded.output
        alias = runner.invoke(
            app,
            ["author", "alias", "Huilian Sophie Qiu", "--add", "Huilian Qiu", "--add", "邱惠莲"],
            env=env,
        )
        assert alias.exit_code == 0, alias.output
        assert alias.output.strip() == "Huilian Sophie Qiu（Huilian Qiu / 邱惠莲）"
        listed = runner.invoke(app, ["author", "list"], env=env)
        assert "Huilian Sophie Qiu（Huilian Qiu / 邱惠莲）" in listed.output
        again = runner.invoke(
            app,
            ["upload", "inbox", "--author", "邱惠莲", "--topic", "1"],
            env=env,
        )
        assert again.exit_code == 0, again.output
        assert "上传 0 篇，跳过 2 篇。" in again.output


def test_cli_upload_prompts_for_author_and_topic():
    with runner.isolated_filesystem():
        inbox = Path("inbox")
        inbox.mkdir()
        (inbox / "Gamma Paper.txt").write_text("gamma", encoding="utf-8")
        env = {
            "AUTOPAPER_CONFIG": str(Path("missing.yaml").resolve()),
            "AUTOPAPER_DATA_DIR": "data",
            "AUTOPAPER_DISABLE_LLM": "1",
            "OPENAI_API_KEY": "",
        }
        result = runner.invoke(app, ["upload", "inbox"], input="Huilian Qiu\nreading group\n", env=env)
        assert result.exit_code == 0, result.output
        assert "上传 1 篇，跳过 0 篇。" in result.output
        assert "Huilian Qiu" in result.output
        assert "reading group" in result.output
