"""Unit tests for Typer CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from local_llm_router.cli.main import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "local-llm-router" in result.stdout


def test_cli_init_config(tmp_path: Path):
    target = tmp_path / "test-router.yaml"
    result = runner.invoke(app, ["init", "-o", str(target)])
    assert result.exit_code == 0
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "length_threshold_tokens" in content
    assert "providers:" in content


def test_cli_check_command():
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 0
    assert "Provider Health Status" in result.stdout


def test_cli_cache_commands(tmp_path: Path):
    # Test cache info
    info_res = runner.invoke(app, ["cache", "info"])
    assert info_res.exit_code == 0
    assert "SQLite Cache Details" in info_res.stdout

    # Test cache clear
    clear_res = runner.invoke(app, ["cache", "clear", "--force"])
    assert clear_res.exit_code == 0
    assert "Successfully cleared" in clear_res.stdout


def test_cli_stats_offline():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Statistics" in result.stdout or "Metrics" in result.stdout
