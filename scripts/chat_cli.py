#!/usr/bin/env python3
"""
Chat CLI — Interactive terminal chatbot for car brochure queries.

Usage:
    python scripts/chat_cli.py

Commands:
    /reset   — Clear conversation history
    /mode    — Show current search mode
    /stats   — Show system stats
    /quit    — Exit the chatbot
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from config import get_settings
from src.chat.engine import ChatEngine

console = Console()


def print_banner():
    """Print the startup banner."""
    banner = Text()
    banner.append("🚗 ", style="bold")
    banner.append("Car Brochure Search System", style="bold cyan")
    banner.append(" — AI Car Advisor\n", style="dim")
    banner.append(f"Search Mode: {get_settings().search_mode.value}", style="yellow")
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))
    console.print()
    console.print("[dim]Ask me anything about cars! Type [bold]/help[/bold] for commands.[/dim]")
    console.print()


def print_help():
    """Print available commands."""
    console.print(Panel(
        "[bold]/reset[/bold]    — Clear conversation history\n"
        "[bold]/mode[/bold]     — Show current search mode\n"
        "[bold]/stats[/bold]    — Show system statistics\n"
        "[bold]/help[/bold]     — Show this help message\n"
        "[bold]/quit[/bold]     — Exit the chatbot",
        title="Commands",
        border_style="blue",
    ))


def main():
    print_banner()

    try:
        engine = ChatEngine()
    except Exception as e:
        console.print(f"[red]Failed to initialize chat engine: {e}[/red]")
        console.print("[dim]Make sure Elasticsearch is running and .env is configured.[/dim]")
        sys.exit(1)

    console.print("[green]✓ Chat engine ready![/green]\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/quit" or cmd == "/exit":
                console.print("[dim]Goodbye! 👋[/dim]")
                break
            elif cmd == "/reset":
                engine.reset_conversation()
                console.print("[yellow]Conversation history cleared.[/yellow]\n")
                continue
            elif cmd == "/mode":
                console.print(f"[yellow]Search mode: {engine.search_mode}[/yellow]\n")
                continue
            elif cmd == "/stats":
                console.print(f"[yellow]Mode: {engine.search_mode}[/yellow]")
                console.print(f"[yellow]History: {engine.history_length} messages[/yellow]\n")
                continue
            elif cmd == "/help":
                print_help()
                continue
            else:
                console.print(f"[red]Unknown command: {user_input}[/red]")
                console.print("[dim]Type /help for available commands.[/dim]\n")
                continue

        # Process the query
        console.print()
        with console.status("[bold cyan]Searching brochures & generating answer...[/bold cyan]"):
            try:
                response = engine.chat(user_input)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")
                continue

        # Display answer
        console.print("[bold green]Advisor:[/bold green]")
        console.print(Markdown(response.answer))

        # Display sources
        if response.sources:
            source_text = " | ".join(
                f"{s['car']} (p.{s['page']})" for s in response.sources[:5]
            )
            console.print(f"\n[dim]Sources: {source_text}[/dim]")

        console.print()


if __name__ == "__main__":
    main()
