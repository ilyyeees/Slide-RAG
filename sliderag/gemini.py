#!/usr/bin/env python3
"""Gemini browser-backed textbook generation for SlideRAG."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from .browser_client import GeminiBrowserClient
from .common import BaseTextbookAgent, SlideData, configure_logging, console

logger = logging.getLogger("sliderag.gemini")


class GeminiTextbookAgent(BaseTextbookAgent):
    """Rolling-context RAG agent using Gemini through browser automation."""

    SYSTEM_PROMPT = """You are an expert textbook author writing a comprehensive, deep-dive chapter in LaTeX format.

CRITICAL ANTI-REPETITION RULES:
1. NEVER repeat a section title that has already been written (check the "ALREADY COVERED" list)
2. NEVER re-explain a concept/definition that was already defined
3. If a topic was covered before, reference it briefly ("As discussed in Section X...") instead of re-explaining
4. Each section must have a UNIQUE title - do not write "Rationality" twice
5. Do NOT reference figures, sources, or sections that don't exist (NO "Source 1", "Figure X", etc.)

SLIDE TYPE HANDLING:
- If a slide is just a TITLE or OUTLINE (1-2 phrases, no bullet points): Write only a brief introduction paragraph (2-3 sentences) setting up what comes next. Do NOT expand into full sections.
- If a slide has SUBSTANTIVE content (bullet points, definitions): Expand fully into comprehensive explanations.

CONTENT PRINCIPLES:
1. EXHAUSTIVE COVERAGE: Expand bullet points into complete, rigorous explanations
2. NARRATIVE FLOW: Create smooth transitions between slides. Link concepts logically
3. INTUITION FIRST: Explain concepts simply before formal equations
4. THE "WHY" AND "HOW": Explain why techniques are needed, not just what they are
5. LaTeX CONVENTIONS: Use the provided custom environments (definition, theorem, example, keyconcept, historicalnote, philosophybox, etc.)
6. CONTINUITY: Your output continues previously generated text. Maintain consistent style
7. DEPTH: Each substantive slide should generate at least 1-2 pages of content

OUTPUT RULES:
- Output ONLY valid LaTeX content - nothing else. Do not output conversational text like "Here is the content...".
- NO markdown code blocks
- NO \\documentclass, \\begin{document}, or \\end{document}
- NO references to non-existent figures or sources
- NO meta-commentary
- Do NOT wrap output in code fences"""

    def __init__(
        self,
        slides_json: str,
        db_path: str = "./chroma_db",
        collection_name: str = "textbook_chunks",
        output_path: str = "output.tex",
        batch_size: int = 5,
        rolling_context_chars: int = 3000,
        retrieval_top_k: int = 4,
        resume: bool = True,
        browser_profile: str = "./gemini_browser_profile",
        headless: bool = False,
        response_timeout: int = 600,
        inter_request_delay: int = 15,
    ):
        super().__init__(
            slides_json=slides_json,
            db_path=db_path,
            collection_name=collection_name,
            output_path=output_path,
            rolling_context_chars=rolling_context_chars,
            retrieval_top_k=retrieval_top_k,
            resume=resume,
        )
        self.batch_size = batch_size
        self.browser_client = GeminiBrowserClient(
            profile_dir=browser_profile,
            headless=headless,
            response_timeout=response_timeout,
            inter_request_delay=inter_request_delay,
        )

    def _make_batches(self, slides: List[SlideData]) -> List[List[SlideData]]:
        return [slides[index : index + self.batch_size] for index in range(0, len(slides), self.batch_size)]

    def _is_batch_completed(self, batch: List[SlideData]) -> bool:
        return self.progress.is_batch_completed(slide.slide_number for slide in batch)

    def _mark_batch_completed(self, batch: List[SlideData]) -> None:
        self.progress.mark_batch_completed(slide.slide_number for slide in batch)

    def _build_batch_prompt(self, batch: List[SlideData], retrieved_chunks: List[str]) -> str:
        parts = ["=== ROLE AND RULES ===", self.SYSTEM_PROMPT, ""]

        chapter_summary = self.chapter_state.get_context_summary()
        if chapter_summary:
            parts.extend(
                [
                    "=== ALREADY COVERED (DO NOT REPEAT THESE) ===",
                    chapter_summary,
                    "",
                ]
            )

        rolling = self._get_rolling_context()
        if rolling:
            parts.extend(
                [
                    "=== PREVIOUS OUTPUT (continue from here, maintain flow) ===",
                    rolling,
                    "",
                ]
            )

        if retrieved_chunks:
            parts.append("=== REFERENCE MATERIAL (use for accuracy, do NOT cite as 'Source X') ===")
            for chunk in retrieved_chunks:
                parts.append(chunk[:1500])
                parts.append("---")
            parts.append("")

        start_slide = batch[0].slide_number
        end_slide = batch[-1].slide_number
        parts.append(f"=== SLIDES TO EXPAND (batch: slides {start_slide}-{end_slide}) ===")
        parts.append("")

        for slide in batch:
            parts.append(f"--- Slide {slide.slide_number}: {slide.title} ---")
            if self._is_title_or_outline_slide(slide):
                parts.append("[TITLE/OUTLINE SLIDE - write only a brief intro paragraph]")
            else:
                parts.append(slide.content)
            parts.append("")

        parts.append("=== INSTRUCTIONS ===")
        parts.append(
            f"Generate comprehensive LaTeX content that expands ALL {len(batch)} slides above into cohesive textbook sections."
        )
        parts.append("1. Expand each substantive slide into full explanations")
        parts.append("2. Use reference material for accuracy without citing 'Source X'")
        parts.append("3. Maintain smooth narrative flow from the previous output")
        parts.append("4. Use appropriate LaTeX environments")
        parts.append("5. Do NOT repeat any section or definition from 'ALREADY COVERED'")
        parts.append("6. Title and outline slides should get only a brief transitional paragraph")
        parts.append("7. Output ONLY valid LaTeX content")
        parts.append("")
        parts.append("Begin generating LaTeX content now:")
        return "\n".join(parts)

    def _process_batch(self, batch: List[SlideData], batch_index: int) -> bool:
        slide_numbers = [slide.slide_number for slide in batch]
        console.print(
            f"\n[bold cyan]batch {batch_index + 1}:[/] slides {slide_numbers[0]}-{slide_numbers[-1]}"
        )
        for slide in batch:
            console.print(f"  [dim]slide {slide.slide_number}[/] {slide.title[:60]}")

        try:
            queries = [slide.get_query() for slide in batch]
            chunks = self.retriever.retrieve_for_batch(queries, self.retrieval_top_k)
            console.print(
                f"  [dim]retrieved [bold]{len(chunks)}[/] unique chunks from textbook[/dim]"
            )

            prompt = self._build_batch_prompt(batch, chunks)
            console.print(f"  [dim]prompt size: {len(prompt):,} chars[/dim]")

            generated = self.browser_client.send_prompt(prompt)
            if not generated.strip() or len(generated.split()) <= 5:
                raise RuntimeError("Gemini returned an empty or too-short response")

            generated = self._clean_latex_output(generated)
            generated = self._remove_duplicate_sections(generated)
            self.chapter_state.extract_from_content(generated)
            self._append_to_output(generated)
            self._update_rolling_context(generated)
            self._mark_batch_completed(batch)
            console.print(
                f"  [green]✓ batch {batch_index + 1} complete ({len(generated):,} chars)[/green]"
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to process batch %s (slides %s-%s): %s",
                batch_index + 1,
                slide_numbers[0],
                slide_numbers[-1],
                exc,
            )
            console.print(f"  [red]✗ batch {batch_index + 1} failed: {exc}[/red]")
            return False

    def run(self) -> None:
        console.print("\n[bold green]===== textbook generation agent (gemini) =====[/]\n")
        console.print("[bold]step 1:[/] launching browser and connecting to Gemini...")
        try:
            self.browser_client.initialize()
        except Exception as exc:
            console.print(f"[bold red]error:[/] could not initialize browser: {exc}")
            console.print(
                "[dim]install playwright and chromium: pip install playwright && playwright install chromium[/dim]"
            )
            return

        try:
            console.print("[green]✓ browser connected[/green]")
            console.print("[bold]step 2:[/] initializing output file...")
            self._initialize_output_file()

            all_batches = self._make_batches(self.slides)
            batches_to_process = [
                (index, batch)
                for index, batch in enumerate(all_batches)
                if not (self.resume and self._is_batch_completed(batch))
            ]
            if not batches_to_process:
                console.print("[yellow]all batches already processed![/yellow]")
                return

            console.print(
                f"[bold]step 3:[/] processing {len(batches_to_process)} batches "
                f"({len(self.slides)} slides, batch size {self.batch_size})"
            )
            console.print(
                f"[dim]chromadb: {self.retriever.get_document_count()} chunks available[/dim]"
            )

            successful = 0
            failed = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "generating textbook...",
                    total=len(batches_to_process),
                )
                for batch_index, batch in batches_to_process:
                    while True:
                        if self._process_batch(batch, batch_index):
                            successful += 1
                            break
                        failed += 1
                        console.print(
                            f"  [yellow]retrying batch {batch_index + 1} in 5 seconds...[/yellow]"
                        )
                        time.sleep(5)
                    progress.update(task, advance=1)

            console.print("\n[bold]step 4:[/] finalizing document...")
            self._finalize_output()
            console.print("\n[bold green]===== generation complete =====[/]\n")
            console.print(f"[bold]batches processed:[/] {successful}/{len(batches_to_process)}")
            if failed > 0:
                console.print(f"[bold red]batch failures:[/] {failed}")
            console.print(f"[bold]output file:[/] {self.output_path}")
            output_size = Path(self.output_path).stat().st_size
            console.print(f"[bold]output size:[/] {output_size:,} bytes")
        finally:
            console.print("\n[dim]closing browser...[/dim]")
            self.browser_client.close()


def create_argparser() -> argparse.ArgumentParser:
    """Create the Gemini CLI parser."""
    parser = argparse.ArgumentParser(
        description="generate LaTeX textbook from slides using Gemini browser automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slides", "-s", type=str, default="slides.json")
    parser.add_argument("--output", "-o", type=str, default="output.tex")
    parser.add_argument("--db-path", "-d", type=str, default="./chroma_db")
    parser.add_argument("--collection", "-c", type=str, default="textbook_chunks")
    parser.add_argument("--batch-size", "-b", type=int, default=5)
    parser.add_argument("--context-chars", type=int, default=3000)
    parser.add_argument("--top-k", "-k", type=int, default=4)
    parser.add_argument("--browser-profile", type=str, default="./gemini_browser_profile")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--response-timeout", type=int, default=600)
    parser.add_argument("--request-delay", type=int, default=15)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--login", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def run_login(browser_profile: str) -> None:
    console.print("\n[bold]opening browser for Google login...[/]\n")
    client = GeminiBrowserClient(profile_dir=browser_profile, headless=False)
    client.initialize()
    console.print("\n[green]✓ login complete[/]")
    console.print(f"[dim]profile directory: {browser_profile}[/dim]")
    client.close()


def run_diagnose(browser_profile: str) -> None:
    console.print("\n[bold]running browser diagnostics...[/]\n")
    client = GeminiBrowserClient(profile_dir=browser_profile, headless=False)
    client.initialize()
    client.diagnose()
    input("\nPress ENTER to close the browser...")
    client.close()


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for the Gemini backend."""
    parser = create_argparser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose, extra_loggers=["sliderag.browser_client"])

    if args.login:
        run_login(args.browser_profile)
        return
    if args.diagnose:
        run_diagnose(args.browser_profile)
        return

    if not Path(args.slides).exists():
        console.print(f"[bold red]error:[/] slides json not found: {args.slides}")
        console.print("[dim]run parse_slides.py first to create slides.json[/dim]")
        sys.exit(1)
    if not Path(args.db_path).exists():
        console.print(f"[bold red]error:[/] chromadb not found at: {args.db_path}")
        console.print("[dim]run setup_rag.py first to create the vector store[/dim]")
        sys.exit(1)

    try:
        agent = GeminiTextbookAgent(
            slides_json=args.slides,
            db_path=args.db_path,
            collection_name=args.collection,
            output_path=args.output,
            batch_size=args.batch_size,
            rolling_context_chars=args.context_chars,
            retrieval_top_k=args.top_k,
            resume=not args.no_resume,
            browser_profile=args.browser_profile,
            headless=args.headless,
            response_timeout=args.response_timeout,
            inter_request_delay=args.request_delay,
        )
        agent.run()
    except FileNotFoundError as exc:
        console.print(f"[bold red]file error:[/] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]generation interrupted by user[/yellow]")
        console.print("[dim]progress saved, run again to resume[/dim]")
        sys.exit(0)
    except Exception as exc:
        console.print(f"[bold red]error:[/] {exc}")
        logger.exception("Generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
