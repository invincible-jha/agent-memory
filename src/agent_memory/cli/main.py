"""CLI entry point for agent-memory.

Invoked as::

    agent-memory [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_memory.cli.main
"""
from __future__ import annotations

import json
import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ---------------------------------------------------------------------------
# Global state: shared UnifiedMemory instance (lazy-initialised per command)
# ---------------------------------------------------------------------------


def _get_memory(db_path: str, backend: str) -> object:
    """Build and return a UnifiedMemory instance based on CLI options."""
    from agent_memory.unified.config import MemoryConfig
    from agent_memory.unified.memory import UnifiedMemory

    config = MemoryConfig(
        storage_backend=backend,  # type: ignore[arg-type]
        sqlite_path=db_path,
    )
    return UnifiedMemory(config=config)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
@click.option(
    "--db",
    "db_path",
    default="agent_memory.db",
    envvar="AGENT_MEMORY_DB",
    show_default=True,
    help="Path to the SQLite database file.",
)
@click.option(
    "--backend",
    default="sqlite",
    type=click.Choice(["sqlite", "memory"], case_sensitive=False),
    show_default=True,
    envvar="AGENT_MEMORY_BACKEND",
    help="Storage backend to use.",
)
@click.pass_context
def cli(ctx: click.Context, db_path: str, backend: str) -> None:
    """Agent memory and knowledge management with 4-layer cognitive architecture."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    ctx.obj["backend"] = backend


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------


@cli.command(name="version")
def version_command() -> None:
    """Show detailed version information."""
    from agent_memory import __version__

    console.print(f"[bold]agent-memory[/bold] v{__version__}")


# ---------------------------------------------------------------------------
# plugins command
# ---------------------------------------------------------------------------


@cli.command(name="plugins")
def plugins_command() -> None:
    """List all registered plugins loaded from entry-points."""
    console.print("[bold]Registered plugins:[/bold]")
    console.print("  (No plugins registered. Install a plugin package to see entries here.)")


# ---------------------------------------------------------------------------
# memory store
# ---------------------------------------------------------------------------


@cli.command(name="store")
@click.argument("content")
@click.option(
    "--layer",
    default="episodic",
    type=click.Choice(["working", "episodic", "semantic", "procedural"], case_sensitive=False),
    show_default=True,
    help="Cognitive layer to store the memory in.",
)
@click.option(
    "--source",
    default="user_input",
    type=click.Choice(
        ["user_input", "tool_output", "agent_inference", "document", "external_api"],
        case_sensitive=False,
    ),
    show_default=True,
    help="Origin source of this memory.",
)
@click.option(
    "--importance",
    type=float,
    default=None,
    help="Override importance score (0.0–1.0). Auto-scored if not provided.",
)
@click.option(
    "--safety-critical",
    is_flag=True,
    default=False,
    help="Mark entry as safety-critical (never decayed or deleted).",
)
@click.option(
    "--meta",
    "metadata_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Metadata key=value pairs (repeatable).",
)
@click.pass_context
def store_command(
    ctx: click.Context,
    content: str,
    layer: str,
    source: str,
    importance: Optional[float],
    safety_critical: bool,
    metadata_pairs: tuple[str, ...],
) -> None:
    """Store a memory CONTENT string into the specified layer."""
    from agent_memory.memory.types import MemoryEntry, MemoryLayer, MemorySource

    memory = _get_memory(ctx.obj["db_path"], ctx.obj["backend"])

    metadata: dict[str, str] = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            console.print(f"[red]Invalid metadata pair {pair!r} — expected KEY=VALUE[/red]")
            sys.exit(1)
        key, _, value = pair.partition("=")
        metadata[key.strip()] = value.strip()

    entry = MemoryEntry(
        content=content,
        layer=MemoryLayer(layer),
        source=MemorySource(source),
        safety_critical=safety_critical,
        metadata=metadata,
    )
    if importance is not None:
        entry = entry.model_copy(update={"importance_score": max(0.0, min(1.0, importance))})

    stored = memory.store(entry)  # type: ignore[attr-defined]
    console.print(
        f"[green]Stored[/green] memory [bold]{stored.memory_id}[/bold] "
        f"in layer [cyan]{stored.layer.value}[/cyan] "
        f"(importance={stored.importance_score:.3f})"
    )


# ---------------------------------------------------------------------------
# memory recall
# ---------------------------------------------------------------------------


@cli.command(name="recall")
@click.argument("memory_id")
@click.option("--json-output", "json_output", is_flag=True, default=False, help="Output as JSON.")
@click.pass_context
def recall_command(ctx: click.Context, memory_id: str, json_output: bool) -> None:
    """Retrieve and display a memory entry by MEMORY_ID."""
    memory = _get_memory(ctx.obj["db_path"], ctx.obj["backend"])
    entry = memory.recall(memory_id)  # type: ignore[attr-defined]

    if entry is None:
        console.print(f"[yellow]No entry found for ID:[/yellow] {memory_id}")
        sys.exit(1)

    if json_output:
        click.echo(entry.model_dump_json(indent=2))
        return

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")

    table.add_row("ID", entry.memory_id)
    table.add_row("Layer", entry.layer.value)
    table.add_row("Source", entry.source.value)
    table.add_row("Importance", f"{entry.importance_score:.4f}")
    table.add_row("Freshness", f"{entry.freshness_score:.4f}")
    table.add_row("Composite", f"{entry.composite_score:.4f}")
    table.add_row("Safety Critical", str(entry.safety_critical))
    table.add_row("Created", entry.created_at.isoformat())
    table.add_row("Last Accessed", entry.last_accessed.isoformat())
    table.add_row("Access Count", str(entry.access_count))
    table.add_row("Content", entry.content)
    if entry.metadata:
        for key, value in entry.metadata.items():
            table.add_row(f"  meta.{key}", value)

    console.print(table)


# ---------------------------------------------------------------------------
# memory search
# ---------------------------------------------------------------------------


@cli.command(name="search")
@click.argument("query")
@click.option(
    "--layer",
    default=None,
    type=click.Choice(["working", "episodic", "semantic", "procedural"], case_sensitive=False),
    help="Restrict search to a single layer.",
)
@click.option("--limit", default=10, show_default=True, help="Maximum results to display.")
@click.option("--json-output", "json_output", is_flag=True, default=False, help="Output as JSON.")
@click.pass_context
def search_command(
    ctx: click.Context,
    query: str,
    layer: Optional[str],
    limit: int,
    json_output: bool,
) -> None:
    """Search memory entries matching QUERY."""
    from agent_memory.memory.types import MemoryLayer

    memory = _get_memory(ctx.obj["db_path"], ctx.obj["backend"])
    layer_enum = MemoryLayer(layer) if layer else None
    results = memory.search(query=query, layer=layer_enum, limit=limit)  # type: ignore[attr-defined]

    if json_output:
        click.echo(
            json.dumps([json.loads(e.model_dump_json()) for e in results], indent=2)
        )
        return

    if not results:
        console.print(f"[yellow]No results for:[/yellow] {query!r}")
        return

    table = Table(
        "Rank",
        "ID",
        "Layer",
        "Importance",
        "Freshness",
        "Content Preview",
        box=box.SIMPLE_HEAD,
        title=f"Search results for {query!r}",
    )
    for rank, entry in enumerate(results, start=1):
        preview = entry.content[:60] + ("..." if len(entry.content) > 60 else "")
        table.add_row(
            str(rank),
            entry.memory_id[:12] + "...",
            entry.layer.value,
            f"{entry.importance_score:.3f}",
            f"{entry.freshness_score:.3f}",
            preview,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# memory stats
# ---------------------------------------------------------------------------


@cli.command(name="stats")
@click.option("--json-output", "json_output", is_flag=True, default=False, help="Output as JSON.")
@click.pass_context
def stats_command(ctx: click.Context, json_output: bool) -> None:
    """Show aggregate memory statistics."""
    memory = _get_memory(ctx.obj["db_path"], ctx.obj["backend"])
    stats = memory.stats()  # type: ignore[attr-defined]

    if json_output:
        click.echo(json.dumps(stats, indent=2))
        return

    table = Table("Metric", "Value", box=box.SIMPLE, show_header=True, title="Memory Stats")
    layer_counts = stats.get("layer_counts", {})
    for layer_name, count in layer_counts.items():  # type: ignore[union-attr]
        table.add_row(f"  {layer_name}", str(count))
    table.add_row("Total (in-process)", str(stats.get("total_in_process", 0)))
    table.add_row("Total (storage)", str(stats.get("total_in_storage", 0)))
    table.add_row("Store calls", str(stats.get("store_calls", 0)))
    table.add_row("Working utilisation", str(stats.get("working_utilisation", 0)))
    table.add_row("Storage backend", str(stats.get("storage_backend", "")))
    table.add_row("Decay function", str(stats.get("decay_function", "")))

    console.print(table)


# ---------------------------------------------------------------------------
# memory consolidate
# ---------------------------------------------------------------------------


@cli.command(name="consolidate")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Identify candidates without actually deleting them.",
)
@click.pass_context
def consolidate_command(ctx: click.Context, dry_run: bool) -> None:
    """Run a garbage-collection pass to remove stale, low-importance entries."""
    from agent_memory.importance.gc import MemoryGarbageCollector

    memory = _get_memory(ctx.obj["db_path"], ctx.obj["backend"])

    if dry_run:
        from agent_memory.memory.types import MemoryLayer

        console.print("[dim]Dry-run mode — no entries will be deleted.[/dim]")
        total_candidates = 0
        for layer in MemoryLayer:
            layer_store = memory._layer_store(layer)  # type: ignore[attr-defined]
            gc = MemoryGarbageCollector(store=layer_store)
            candidates = gc.candidates()
            if candidates:
                console.print(
                    f"  [yellow]{len(candidates)}[/yellow] candidate(s) in "
                    f"[cyan]{layer.value}[/cyan]"
                )
                total_candidates += len(candidates)
        console.print(f"Total candidates: [yellow]{total_candidates}[/yellow]")
        return

    result = memory.consolidate()  # type: ignore[attr-defined]
    console.print(
        f"[green]Consolidation complete.[/green] "
        f"Collected: [red]{result['collected']}[/red] | "
        f"Retained: [green]{result['retained']}[/green]"
    )


# ---------------------------------------------------------------------------
# memory scan-contradictions
# ---------------------------------------------------------------------------


@cli.command(name="scan-contradictions")
@click.option(
    "--scope",
    default="all",
    type=click.Choice(
        ["all", "working", "episodic", "semantic", "procedural", "high_importance"],
        case_sensitive=False,
    ),
    show_default=True,
    help="Which entries to scan.",
)
@click.option(
    "--max",
    "max_contradictions",
    default=0,
    show_default=True,
    help="Stop after this many contradictions (0 = no limit).",
)
@click.option("--json-output", "json_output", is_flag=True, default=False, help="Output as JSON.")
@click.pass_context
def scan_contradictions_command(
    ctx: click.Context,
    scope: str,
    max_contradictions: int,
    json_output: bool,
) -> None:
    """Scan memory for contradicting entries and report findings."""
    from agent_memory.contradiction.scanner import ContradictionScanner, ScanScope
    from agent_memory.memory.types import MemoryLayer
    from agent_memory.unified.memory import UnifiedMemory

    memory: UnifiedMemory = _get_memory(ctx.obj["db_path"], ctx.obj["backend"])  # type: ignore[assignment]
    scanner = ContradictionScanner(max_contradictions=max_contradictions)
    scan_scope = ScanScope(scope)

    # Load entries from durable storage, filtered by scope
    if scan_scope == ScanScope.ALL:
        entries = memory.storage.load_all(limit=5000)
    elif scan_scope == ScanScope.HIGH_IMPORTANCE:
        all_entries = memory.storage.load_all(limit=5000)
        entries = [e for e in all_entries if e.importance_score >= 0.6]
    else:
        # Map scope to layer
        layer_map = {
            ScanScope.WORKING: MemoryLayer.WORKING,
            ScanScope.EPISODIC: MemoryLayer.EPISODIC,
            ScanScope.SEMANTIC: MemoryLayer.SEMANTIC,
            ScanScope.PROCEDURAL: MemoryLayer.PROCEDURAL,
        }
        target_layer = layer_map.get(scan_scope)
        entries = memory.storage.load_all(layer=target_layer, limit=5000)

    result = scanner.scan_entries(entries)

    summary = result.summary()

    if json_output:
        output = {
            "summary": summary,
            "pairs": [
                {
                    "entry_a_id": p.entry_a_id,
                    "entry_b_id": p.entry_b_id,
                    "similarity": p.similarity_score,
                    "conflict": p.conflict_description,
                    "entry_a_preview": p.entry_a_content[:80],
                    "entry_b_preview": p.entry_b_content[:80],
                }
                for p in result.contradiction_pairs
            ],
        }
        click.echo(json.dumps(output, indent=2))
        return

    console.print(
        f"\n[bold]Contradiction Scan Results[/bold]\n"
        f"  Scanned: {summary['scanned_count']} entries\n"
        f"  Duration: {summary['scan_duration_seconds']:.3f}s\n"
        f"  Early terminated: {summary['early_terminated']}"
    )

    if not result.contradiction_pairs:
        console.print("[green]No contradictions found.[/green]")
        return

    table = Table(
        "#",
        "Entry A (preview)",
        "Entry B (preview)",
        "Similarity",
        "Conflict",
        box=box.SIMPLE_HEAD,
        title=f"{result.contradiction_count} Contradiction(s) Found",
    )
    for index, pair in enumerate(result.contradiction_pairs, start=1):
        table.add_row(
            str(index),
            pair.entry_a_content[:40] + ("..." if len(pair.entry_a_content) > 40 else ""),
            pair.entry_b_content[:40] + ("..." if len(pair.entry_b_content) > 40 else ""),
            f"{pair.similarity_score:.3f}",
            pair.conflict_description[:50],
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
