"""CLI check command: verify provider availability."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from local_llm_router.config import load_config
from local_llm_router.providers.registry import build_registry

console = Console()


def check_command(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to router.yaml"),
) -> None:
    """Check health and reachability of all configured providers."""
    cfg = load_config(config_file)
    registry = build_registry(cfg.providers)

    async def _run_checks():
        results = []
        for p in registry.list_all():
            try:
                healthy = await p.health()
                results.append((p.name, p.default_model, healthy, None))
            except Exception as e:
                results.append((p.name, p.default_model, False, str(e)))
        await registry.close_all()
        return results

    results = asyncio.run(_run_checks())

    table = Table(title="Provider Health Status", show_header=True)
    table.add_column("Provider Name", style="cyan")
    table.add_column("Default Model", style="blue")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    for name, model, healthy, err in results:
        status_text = "[green]ONLINE / HEALTHY[/green]" if healthy else "[red]OFFLINE[/red]"
        details = err or ("Responsive" if healthy else "Unreachable or returned error")
        table.add_row(name, model, status_text, details)

    console.print(table)
