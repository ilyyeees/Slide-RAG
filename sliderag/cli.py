"""unified cli for sliderag."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass(frozen=True)
class CommandSpec:
    """metadata for a top-level cli command."""

    name: str
    accent: str
    summary: str
    detail: str
    examples: List[str]


COMMAND_SPECS: Dict[str, CommandSpec] = {
    "parse-slides": CommandSpec(
        name="parse-slides",
        accent="bright_cyan",
        summary="turn a slide deck pdf into structured json",
        detail="this is the slide extraction module. it reads a lecture deck, preserves structure, and writes the json that the generators consume.",
        examples=[
            "python -m sliderag parse-slides -- --pdf data/input/slides.pdf --output runs/course/slides.json",
            "python parse_slides.py --pdf slides.pdf --output slides.json",
        ],
    ),
    "setup-rag": CommandSpec(
        name="setup-rag",
        accent="bright_blue",
        summary="index a textbook into chromadb",
        detail="this is the ingestion module. it chunks the textbook, embeds the chunks, and writes the vector index used by both backends.",
        examples=[
            "python -m sliderag setup-rag -- --pdf data/input/textbook.pdf --db-path data/indexes/chroma_db",
            "python setup_rag.py --pdf textbook.pdf --db-path ./chroma_db",
        ],
    ),
    "generate": CommandSpec(
        name="generate",
        accent="bright_magenta",
        summary="run textbook generation with ollama or gemini",
        detail="this is the generation front door. choose a backend, then pass backend-specific flags through after the separator.",
        examples=[
            "python -m sliderag generate --backend ollama -- --slides runs/course/slides.json --model deepseek-r1:32b",
            "python -m sliderag generate --backend gemini -- --slides runs/course/slides.json --batch-size 3",
        ],
    ),
    "gemini-login": CommandSpec(
        name="gemini-login",
        accent="medium_purple",
        summary="open the persistent gemini browser profile and log in",
        detail="use this once to attach your google session to the saved browser profile used by the gemini backend.",
        examples=[
            "python -m sliderag gemini-login",
            "python gemini_agent.py --login",
        ],
    ),
    "gemini-diagnose": CommandSpec(
        name="gemini-diagnose",
        accent="gold1",
        summary="inspect gemini selectors when the ui changes",
        detail="this opens the browser and prints the currently detectable elements so you can update selectors if google changes the page.",
        examples=[
            "python -m sliderag gemini-diagnose",
            "python gemini_agent.py --diagnose",
        ],
    ),
}


def create_argparser() -> argparse.ArgumentParser:
    """create the top-level parser."""
    parser = argparse.ArgumentParser(
        description="sliderag unified command line interface",
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command")

    parse_subparser = subparsers.add_parser("parse-slides", add_help=False)
    parse_subparser.add_argument("args", nargs=argparse.REMAINDER)

    rag_subparser = subparsers.add_parser("setup-rag", add_help=False)
    rag_subparser.add_argument("args", nargs=argparse.REMAINDER)

    generate_subparser = subparsers.add_parser("generate", add_help=False)
    generate_subparser.add_argument(
        "--backend",
        choices=["ollama", "gemini"],
        required=False,
    )
    generate_subparser.add_argument("args", nargs=argparse.REMAINDER)

    login_subparser = subparsers.add_parser("gemini-login", add_help=False)
    login_subparser.add_argument("args", nargs=argparse.REMAINDER)

    diagnose_subparser = subparsers.add_parser("gemini-diagnose", add_help=False)
    diagnose_subparser.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def _strip_separator(args: List[str]) -> List[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _render_home() -> None:
    title = Text("sliderag", style="bold white")
    subtitle = Text("one pipeline, two backends, one clean front door", style="bold bright_cyan")
    intro = Text.assemble(
        title,
        "\n",
        subtitle,
        "\n\n",
        ("shared modules live in one package now. use this cli to parse slides, build rag indexes, and run either generator.", "white"),
    )
    console.print(
        Panel(
            intro,
            border_style="bright_blue",
            padding=(1, 2),
            title="project hub",
            title_align="left",
        )
    )

    command_table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold white",
        expand=True,
        pad_edge=False,
    )
    command_table.add_column("command", style="bold bright_white", width=18)
    command_table.add_column("what it does", style="white")
    command_table.add_column("best for", style="dim")

    command_table.add_row(
        "parse-slides",
        COMMAND_SPECS["parse-slides"].summary,
        "extracting structured slide content",
    )
    command_table.add_row(
        "setup-rag",
        COMMAND_SPECS["setup-rag"].summary,
        "building the shared textbook index",
    )
    command_table.add_row(
        "generate",
        COMMAND_SPECS["generate"].summary,
        "running the actual authoring flow",
    )
    command_table.add_row(
        "gemini-login",
        COMMAND_SPECS["gemini-login"].summary,
        "initial gemini session setup",
    )
    command_table.add_row(
        "gemini-diagnose",
        COMMAND_SPECS["gemini-diagnose"].summary,
        "fixing gemini browser selector drift",
    )

    console.print(command_table)
    console.print()

    workflow = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_cyan",
        expand=True,
    )
    workflow.add_column("step", width=8, style="bold white")
    workflow.add_column("command", style="white")
    workflow.add_row(
        "1",
        "python -m sliderag setup-rag -- --pdf data/input/textbook.pdf --db-path data/indexes/chroma_db",
    )
    workflow.add_row(
        "2",
        "python -m sliderag parse-slides -- --pdf data/input/slides.pdf --output runs/course/slides.json",
    )
    workflow.add_row(
        "3a",
        "python -m sliderag generate --backend ollama -- --slides runs/course/slides.json",
    )
    workflow.add_row(
        "3b",
        "python -m sliderag generate --backend gemini -- --slides runs/course/slides.json",
    )
    console.print(Panel(workflow, border_style="bright_cyan", title="default workflow", title_align="left"))

    tips = Text()
    tips.append("notes\n", style="bold bright_magenta")
    tips.append("- use ", style="white")
    tips.append("python -m sliderag help <command>", style="bold white")
    tips.append(" for command-specific guidance\n", style="white")
    tips.append("- use ", style="white")
    tips.append("--", style="bold white")
    tips.append(" before backend flags when you want clean pass-through behavior\n", style="white")
    tips.append("- compatibility wrappers still work: ", style="white")
    tips.append("main_agent.py", style="bold white")
    tips.append(", ", style="white")
    tips.append("parse_slides.py", style="bold white")
    tips.append(", ", style="white")
    tips.append("setup_rag.py", style="bold white")
    console.print(Panel(tips, border_style="bright_magenta", title="usage", title_align="left"))


def _render_command_help(command: str) -> None:
    spec = COMMAND_SPECS.get(command)
    if spec is None:
        console.print(f"[bold red]unknown command:[/] {command}")
        console.print()
        _render_home()
        raise SystemExit(1)

    console.print(Rule(style=spec.accent))
    body = Text()
    body.append(f"{spec.name}\n", style=f"bold {spec.accent}")
    body.append(spec.summary + "\n\n", style="white")
    body.append(spec.detail + "\n\n", style="dim")
    body.append("examples\n", style="bold white")
    for example in spec.examples:
        body.append(f"{example}\n", style="bright_white")

    if command == "generate":
        body.append("\nbackend selection\n", style="bold white")
        body.append("- use ", style="white")
        body.append("--backend ollama", style="bright_white")
        body.append(" for local inference through the ollama api\n", style="white")
        body.append("- use ", style="white")
        body.append("--backend gemini", style="bright_white")
        body.append(" for browser-driven generation against gemini\n", style="white")

    console.print(Panel(body, border_style=spec.accent, title="command guide", title_align="left"))
    console.print(Rule(style=spec.accent))


def _render_dispatch_banner(spec: CommandSpec, extra: Optional[str] = None) -> None:
    body = Text()
    body.append(spec.name + "\n", style=f"bold {spec.accent}")
    body.append(spec.summary, style="white")
    if extra:
        body.append("\n\n", style="white")
        body.append(extra, style="dim")

    console.print(
        Panel(
            body,
            border_style=spec.accent,
            title="sliderag",
            title_align="left",
            padding=(1, 2),
        )
    )


def _dispatch_parse_slides(raw_args: List[str]) -> None:
    from . import parse_slides

    _render_dispatch_banner(COMMAND_SPECS["parse-slides"], "forwarding into the slide extraction module")
    parse_slides.main(_strip_separator(raw_args))


def _dispatch_setup_rag(raw_args: List[str]) -> None:
    from . import setup_rag

    _render_dispatch_banner(COMMAND_SPECS["setup-rag"], "forwarding into the rag ingestion module")
    setup_rag.main(_strip_separator(raw_args))


def _dispatch_generate(backend: Optional[str], raw_args: List[str]) -> None:
    if backend is None:
        _render_command_help("generate")
        raise SystemExit(1)

    extra = f"selected backend: {backend}"
    _render_dispatch_banner(COMMAND_SPECS["generate"], extra)
    remaining = _strip_separator(raw_args)

    if backend == "ollama":
        from . import ollama

        ollama.main(remaining)
        return

    from . import gemini

    gemini.main(remaining)


def _dispatch_gemini_login(raw_args: List[str]) -> None:
    from . import gemini

    _render_dispatch_banner(COMMAND_SPECS["gemini-login"], "opening the persistent gemini browser profile")
    gemini.main(["--login", *_strip_separator(raw_args)])


def _dispatch_gemini_diagnose(raw_args: List[str]) -> None:
    from . import gemini

    _render_dispatch_banner(COMMAND_SPECS["gemini-diagnose"], "launching the browser diagnostics view")
    gemini.main(["--diagnose", *_strip_separator(raw_args)])


def main(argv: Optional[List[str]] = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)

    if not args_list:
        _render_home()
        return

    if args_list[0] in {"-h", "--help"}:
        _render_home()
        return

    if args_list[0] == "help":
        if len(args_list) == 1:
            _render_home()
            return
        _render_command_help(args_list[1])
        return

    if len(args_list) >= 2 and args_list[1] in {"-h", "--help"}:
        _render_command_help(args_list[0])
        return

    parser = create_argparser()
    args = parser.parse_args(args_list)

    dispatchers: Dict[str, Callable[..., None]] = {
        "parse-slides": _dispatch_parse_slides,
        "setup-rag": _dispatch_setup_rag,
        "gemini-login": _dispatch_gemini_login,
        "gemini-diagnose": _dispatch_gemini_diagnose,
    }

    if args.command == "generate":
        _dispatch_generate(args.backend, args.args)
        return

    dispatcher = dispatchers.get(args.command)
    if dispatcher is None:
        _render_home()
        raise SystemExit(1)

    dispatcher(args.args)


if __name__ == "__main__":
    main(sys.argv[1:])
