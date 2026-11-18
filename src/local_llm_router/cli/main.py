"""Main CLI entrypoint for local-llm-router."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from local_llm_router import __version__
from local_llm_router.cli.commands.cache import cache_app
from local_llm_router.cli.commands.check import check_command
from local_llm_router.cli.commands.serve import serve_command
from local_llm_router.cli.commands.stats import stats_command

app = typer.Typer(
    name="local-llm-router",
    help="Local LLM/SLM Router & Cache Proxy with OpenAI-compatible API, JSON validation, and fallbacks.",
    add_completion=False,
)
console = Console()

# Register subcommands
app.command("serve", help="Start the router HTTP proxy server.")(serve_command)
app.command("stats", help="Show real-time cache hit-rate, latency, and saved tokens.")(stats_command)
app.command("check", help="Check reachability of configured local and cloud providers.")(check_command)
app.add_typer(cache_app, name="cache", help="Manage local SQLite cache.")


@app.command("init")
def init_config(
    output: str = typer.Option("router.yaml", "--output", "-o", help="Target filename"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite if exists"),
) -> None:
    """Generate a starter router.yaml configuration file."""
    path = Path(output)
    if path.exists() and not force:
        console.print(f"[red]Error:[/red] File {output} already exists. Use --force to overwrite.")
        raise typer.Exit(code=1)

    sample_yaml = """# Local LLM Router Configuration
server:
  host: 127.0.0.1
  port: 8000
  workers: 1
  log_level: info

cache:
  enabled: true
  db_path: data/cache.db
  ttl_seconds: 86400  # 24 hours (0 = never expire)
  max_entries: 10000  # LRU eviction limit
  normalize_whitespace: true

validator:
  enabled: true
  max_retries: 3
  strip_markdown: true

routing:
  default_route: slm
  length_threshold_tokens: 250  # Prompts <= 250 tokens go to SLM, > 250 to Heavy
  slm_target: ollama:llama3.2:1b
  heavy_target: openai:gpt-4o-mini

  tags:
    fast: slm
    code: ollama:qwen2.5-coder
    reasoning: openai:gpt-4o
    heavy: heavy

  aliases:
    slm: ollama:llama3.2:1b
    heavy: openai:gpt-4o-mini
    default: ollama:llama3.2:1b
    fast: ollama:llama3.2:1b

  fallback_chains:
    default:
      - ollama:llama3.2:1b
      - openai:gpt-4o-mini
      - mock:fallback

providers:
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    default_model: llama3.2:1b
    timeout_seconds: 30.0
    models:
      - llama3.2:1b
      - qwen2.5-coder

  - name: llamacpp
    type: llama_cpp
    base_url: http://localhost:8080
    default_model: local-model
    timeout_seconds: 60.0

  - name: openai
    type: openai_compatible
    base_url: https://api.openai.com/v1
    api_key: "${OPENAI_API_KEY}"
    default_model: gpt-4o-mini
    timeout_seconds: 30.0
    models:
      - gpt-4o-mini
      - gpt-4o

  - name: mock
    type: mock
    default_model: mock-model
    models:
      - mock-model
      - fallback
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(sample_yaml.strip() + "\n")

    console.print(f"[bold green]Created sample configuration at {output}[/bold green]")


@app.callback(invoke_without_command=True)
def main_callback(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show application version and exit"
    ),
) -> None:
    if version:
        console.print(f"[bold cyan]local-llm-router[/bold cyan] version [green]{__version__}[/green]")
        raise typer.Exit()


if __name__ == "__main__":
    app()
