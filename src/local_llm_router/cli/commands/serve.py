"""CLI command: serve."""

from __future__ import annotations

from typing import Optional

import typer
import uvicorn
from rich.console import Console

from local_llm_router.config import load_config

console = Console()


def serve_command(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Bind host"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Bind port"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to router.yaml"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of worker processes"),
) -> None:
    """Start the Local LLM Router & Cache Proxy server."""
    cfg = load_config(config_file)
    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port
    worker_count = workers or cfg.server.workers

    console.print(
        f"[bold green]Starting Local LLM Router & Proxy[/bold green] on [cyan]http://{bind_host}:{bind_port}[/cyan]"
    )
    console.print(f"Loaded config with [yellow]{len(cfg.providers)} providers[/yellow]")
    console.print("Press [bold red]Ctrl+C[/bold red] to stop.\n")

    # Run uvicorn
    from local_llm_router.api.app import create_app

    app = create_app(cfg)
    uvicorn.run(
        app,
        host=bind_host,
        port=bind_port,
        workers=worker_count if not reload else 1,
        reload=reload,
        log_level=cfg.server.log_level,
    )
