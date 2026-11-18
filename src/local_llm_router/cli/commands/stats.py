"""CLI command: stats."""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

from local_llm_router.cache.sqlite_cache import SQLiteCache
from local_llm_router.config import load_config

console = Console()


def stats_command(
    url: str = typer.Option("http://127.0.0.1:8000", "--url", "-u", help="URL of running router proxy"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to router.yaml"),
) -> None:
    """Display real-time telemetry: cache hit-rate, latency, saved tokens, and fallbacks."""
    # Attempt to fetch live telemetry from running HTTP server
    try:
        resp = httpx.get(f"{url.rstrip('/')}/stats", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            _render_live_stats(data)
            return
    except Exception:
        pass

    # Fallback to local SQLite cache inspection
    console.print("[dim]Live server not detected; reading directly from local SQLite cache...[/dim]\n")
    cfg = load_config(config_file)
    asyncio.run(_render_cache_direct(cfg))


def _render_live_stats(data: dict) -> None:
    """Render telemetry table from live server stats."""
    table = Table(title="Local LLM Router - Telemetry & Cache Metrics", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Total Requests", str(data.get("total_requests", 0)))
    table.add_row("Successful Requests", f"[green]{data.get('successful_requests', 0)}[/green]")
    table.add_row("Failed Requests", f"[red]{data.get('failed_requests', 0)}[/red]")
    table.add_row("Cache Hits", f"[bold green]{data.get('cache_hits', 0)}[/bold green]")
    table.add_row("Cache Misses", str(data.get("cache_misses", 0)))
    table.add_row("Cache Bypasses", str(data.get("cache_bypasses", 0)))

    hit_rate = data.get("cache_hit_rate_pct", 0.0)
    color = "green" if hit_rate > 50 else ("yellow" if hit_rate > 20 else "white")
    table.add_row("Cache Hit Rate", f"[{color}]{hit_rate:.1f}%[/{color}]")

    table.add_row("Tokens Saved via Cache", f"[bold cyan]{data.get('total_tokens_saved', 0):,}[/bold cyan]")
    table.add_row("Total Prompt Tokens", f"{data.get('total_prompt_tokens', 0):,}")
    table.add_row("Total Completion Tokens", f"{data.get('total_completion_tokens', 0):,}")
    table.add_row("Average Latency", f"{data.get('avg_latency_ms', 0):.2f} ms")
    table.add_row("p95 Latency", f"{data.get('p95_latency_ms', 0):.2f} ms")
    table.add_row("Fallback Events", f"[yellow]{data.get('fallback_events_total', 0)}[/yellow]")
    table.add_row("JSON Validation Retries", str(data.get("validation_retries_total", 0)))

    console.print(table)

    providers = data.get("provider_calls", {})
    if providers:
        p_table = Table(title="Provider Request Breakdown", show_header=True)
        p_table.add_column("Provider", style="cyan")
        p_table.add_column("Calls", style="green")
        for p, count in providers.items():
            p_table.add_row(p, str(count))
        console.print(p_table)


async def _render_cache_direct(cfg) -> None:
    """Render cache stats directly from SQLite database."""
    cache = SQLiteCache(cfg.cache)
    stats = await cache.get_stats()
    await cache.close()

    table = Table(title="Offline Cache Database Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Database File", str(cfg.cache.db_path))
    table.add_row("Total Cached Responses", str(stats.total_entries))
    table.add_row("Total Hit Count", str(stats.hit_count))
    table.add_row("Total Tokens Saved", f"[bold green]{stats.tokens_saved:,}[/bold green]")
    table.add_row("Database Size", f"{stats.size_bytes / 1024:.2f} KB")

    console.print(table)
