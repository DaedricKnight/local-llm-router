"""CLI cache commands: clear and info."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from local_llm_router.cache.sqlite_cache import SQLiteCache
from local_llm_router.config import load_config

cache_app = typer.Typer(help="Manage local SQLite cache.")
console = Console()


@cache_app.command("clear")
def cache_clear(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to router.yaml"),
    force: bool = typer.Option(False, "--force", "-f", help="Do not ask for confirmation"),
) -> None:
    """Clear all records from the SQLite cache database."""
    if not force:
        confirm = typer.confirm("Are you sure you want to purge all cached responses?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            return

    cfg = load_config(config_file)

    async def _do_clear() -> int:
        c = SQLiteCache(cfg.cache)
        count = await c.clear()
        await c.close()
        return count

    deleted = asyncio.run(_do_clear())
    console.print(f"[bold green]Successfully cleared {deleted} cache entries.[/bold green]")


@cache_app.command("info")
def cache_info(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to router.yaml"),
) -> None:
    """Show cache storage parameters and metrics."""
    cfg = load_config(config_file)

    async def _get_info():
        c = SQLiteCache(cfg.cache)
        stats = await c.get_stats()
        await c.close()
        return stats

    stats = asyncio.run(_get_info())

    table = Table(title="SQLite Cache Details")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Cache Enabled", str(cfg.cache.enabled))
    table.add_row("Database Path", cfg.cache.db_path)
    table.add_row("TTL Seconds", f"{cfg.cache.ttl_seconds} s ({cfg.cache.ttl_seconds / 3600:.1f} hours)")
    table.add_row("Max Entries (LRU)", str(cfg.cache.max_entries))
    table.add_row("Total Entries Currently", str(stats.total_entries))
    table.add_row("Recorded Tokens Saved", f"{stats.tokens_saved:,}")
    table.add_row("Disk Size", f"{stats.size_bytes / 1024:.2f} KB")

    console.print(table)
