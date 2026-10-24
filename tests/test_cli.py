from pathlib import Path

import pytest
from typer.testing import CliRunner

from autopaper.cli import app
from autopaper.models import Paper


runner = CliRunner()


def isolated_env(root: Path) -> dict[str, str]:
    return {
        "AUTOPAPER_CONFIG": str(root / "missing.yaml"),
        "AUTOPAPER_DATA_DIR": "data",
        "AUTOPAPER_DISABLE_LLM": "1",
        "OPENAI_API_KEY": "",
    }


def test_topic_collect_report_and_search(monkeypatch: pytest.MonkeyPatch):
    class FakeSource:
        name = "arxiv"

        def search(self, query, *, limit, since_year=None, sort="relevance"):
            return [
                Paper(
                    title="Graph Neural Networks for Time Series Forecasting",
                    abstract="graph neural networks forecast time series",
                    year=2024,
                    doi="10.1000/cli",
                    authors=["Ada Lovelace"],
                    source="arxiv",
                    citation_count=2,
                ),
                Paper(
                    title="Graph Neural Networks for Time Series Forecasting",
                    doi="10.1000/CLI",
                    abstract="duplicate record",
                    source="crossref",
                ),
                Paper(title="Cooking pasta", abstract="water", year=2020, source="pubmed"),
            ]

    monkeypatch.setattr("autopaper.cli.build_sources", lambda names, settings, http: [FakeSource()])
    with runner.isolated_filesystem():
        root = Path.cwd()
        env = isolated_env(root)
        added = runner.invoke(
            app,
            ["topic", "add", "Graph Neural Networks", "--keyword", "graph neural networks", "--schedule", "weekly"],
            env=env,
        )
        assert added.exit_code == 0, added.output
        assert "graph-neural-networks" in added.output

        listed = runner.invoke(app, ["topic", "list"], env=env)
        assert listed.exit_code == 0
        assert "Graph Neural Networks" in listed.output

        edited = runner.invoke(
            app,
            ["topic", "edit", "graph-neural-networks", "--schedule", "daily", "--min-score", "0"],
            env=env,
        )
        assert edited.exit_code == 0, edited.output
        shown = runner.invoke(app, ["topic", "show", "Graph Neural Networks"], env=env)
        assert "schedule: daily" in shown.output
        assert "graph neural networks" in shown.output

        collected = runner.invoke(app, ["collect", "graph-neural-networks"], env=env)
        text = collected.output
        assert collected.exit_code == 0, text
        assert "去重后 2" in text
        assert "过滤后 1" in text

        papers = runner.invoke(app, ["papers", "graph-neural-networks"], env=env)
        assert "Graph Neural Networks for Time Series Forecasting" in papers.output
        found = runner.invoke(app, ["search", "forecast"], env=env)
        assert found.exit_code == 0
        assert "forecast" in found.output.casefold() or "Forecasting" in found.output

        report = runner.invoke(app, ["report", "graph-neural-networks"], env=env)
        report_text = report.output
        assert report.exit_code == 0, report_text
        assert "引用列表" in report_text
        assert "未生成模型综述" in report_text

        reading = runner.invoke(app, ["read", "10.1000/cli"], env=env)
        assert reading.exit_code == 1
        assert "OPENAI_API_KEY" in reading.output

        removed = runner.invoke(app, ["topic", "rm", "graph-neural-networks", "--yes"], env=env)
        assert removed.exit_code == 0
        missing = runner.invoke(app, ["topic", "show", "graph-neural-networks"], env=env)
        assert missing.exit_code == 2


def test_unknown_source_is_rejected():
    with runner.isolated_filesystem():
        env = isolated_env(Path.cwd())
        result = runner.invoke(app, ["topic", "add", "Test", "--source", "not-a-source"], env=env)
        assert result.exit_code == 1
        assert "未知数据源" in result.output


def test_schedule_dry_run_lists_new_daily_topic():
    with runner.isolated_filesystem():
        env = isolated_env(Path.cwd())
        runner.invoke(app, ["topic", "add", "Daily topic", "--schedule", "daily", "--source", "arxiv"], env=env)
        result = runner.invoke(app, ["schedule", "run", "--dry-run"], env=env)
        text = result.output
        assert result.exit_code == 0, text
        assert "到期：daily-topic" in text
