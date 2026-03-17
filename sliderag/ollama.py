#!/usr/bin/env python3
"""
Ollama-backed textbook generation for SlideRAG.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import requests
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from .common import BaseTextbookAgent, SlideData, configure_logging, console

logger = logging.getLogger("sliderag.ollama")


class OllamaClient:
    """HTTP client for the local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1:32b",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_health(self) -> bool:
        """Return True when Ollama is reachable and the requested model exists."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            model_base = self.model.split(":")[0]
            available = any(
                self.model in name or model_base in name for name in model_names
            )
            if not available:
                logger.warning("Model '%s' not found. Available: %s", self.model, model_names)
            return available
        except requests.RequestException as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        max_retries: int = 3,
    ) -> str:
        """Generate content through the Ollama API."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 8192,
            },
        }
        if system:
            payload["system"] = system

        for attempt in range(max_retries):
            try:
                if stream:
                    return self._generate_streaming(payload)
                return self._generate_sync(payload)
            except (requests.RequestException, httpx.HTTPError) as exc:
                logger.warning(
                    "Generation attempt %s failed: %s",
                    attempt + 1,
                    exc,
                )
                if attempt == max_retries - 1:
                    raise
                time.sleep(2**attempt)

        return ""

    def _generate_sync(self, payload: Dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _generate_streaming(self, payload: Dict[str, Any]) -> str:
        payload["stream"] = True
        full_response: List[str] = []
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    full_response.append(data.get("response", ""))
                    if data.get("done", False):
                        break
        return "".join(full_response)


class OllamaTextbookAgent(BaseTextbookAgent):
    """Rolling-context RAG agent using a local Ollama model."""

    SYSTEM_PROMPT = """You are an expert textbook author writing a comprehensive, deep-dive chapter in LaTeX format.

CRITICAL ANTI-REPETITION RULES:
1. NEVER repeat a section title that has already been written (check the "ALREADY COVERED" list)
2. NEVER re-explain a concept/definition that was already defined
3. If a topic was covered before, reference it briefly ("As discussed in Section X...") instead of re-explaining
4. Each section must have a UNIQUE title - do not write "Rationality" twice
5. Do NOT reference figures, sources, or sections that don't exist (NO "Source 1", "Figure X", etc.)

SLIDE TYPE HANDLING:
- If the slide is just a TITLE or OUTLINE (1-2 phrases, no bullet points): Write only a brief introduction paragraph (2-3 sentences) setting up what comes next. Do NOT expand into full sections.
- If the slide has SUBSTANTIVE content (bullet points, definitions): Expand fully into comprehensive explanations.

CONTENT PRINCIPLES:
1. EXHAUSTIVE COVERAGE: Expand bullet points into complete, rigorous explanations
2. NARRATIVE FLOW: Create smooth transitions. Link concepts logically
3. INTUITION FIRST: Explain concepts simply before equations
4. THE "WHY" AND "HOW": Explain why techniques are needed
5. LaTeX CONVENTIONS: Use provided environments (definition, theorem, example, keyconcept, etc.)
6. CONTINUITY: Your output continues previously generated text. Maintain consistent style

OUTPUT RULES:
- Output ONLY valid LaTeX content
- NO markdown code blocks
- NO \\documentclass, \\begin{document}, or \\end{document}
- NO references to non-existent figures or sources"""

    def __init__(
        self,
        slides_json: str,
        db_path: str = "./chroma_db",
        collection_name: str = "textbook_chunks",
        output_path: str = "output.tex",
        model: str = "deepseek-r1:32b",
        ollama_url: str = "http://localhost:11434",
        rolling_context_chars: int = 1000,
        retrieval_top_k: int = 5,
        resume: bool = True,
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
        self.ollama = OllamaClient(ollama_url, model)

    def _build_prompt(self, slide: SlideData, retrieved_chunks: List[str]) -> str:
        parts = []

        chapter_summary = self.chapter_state.get_context_summary()
        if chapter_summary:
            parts.append("=== ALREADY COVERED (DO NOT REPEAT) ===")
            parts.append(chapter_summary)
            parts.append("")

        rolling = self._get_rolling_context()
        if rolling:
            parts.append("=== PREVIOUS OUTPUT (for continuity) ===")
            parts.append(rolling)
            parts.append("")

        if retrieved_chunks:
            parts.append("=== REFERENCE MATERIAL ===")
            for chunk in retrieved_chunks:
                parts.append(chunk[:1500])
                parts.append("---")
            parts.append("")

        parts.append("=== SLIDE CONTENT TO EXPAND ===")
        parts.append(f"Slide {slide.slide_number}: {slide.title}")
        parts.append("")

        is_title_slide = self._is_title_or_outline_slide(slide)
        if is_title_slide:
            parts.append(
                "NOTE: This is a TITLE/OUTLINE slide. Write only a brief introduction (2-3 sentences)."
            )
        else:
            parts.append("Content to cover:")
            parts.append(slide.content)
        parts.append("")

        parts.append("=== INSTRUCTIONS ===")
        if is_title_slide:
            parts.append("Write a brief transitional paragraph introducing the upcoming topic.")
            parts.append("Do NOT create new sections or definitions for title slides.")
        else:
            parts.append("Generate comprehensive LaTeX content that:")
            parts.append("1. Expands each bullet point into full explanations")
            parts.append("2. Uses reference material for accuracy (but don't cite 'Source X')")
            parts.append("3. Maintains smooth flow from the previous output")
            parts.append("4. Uses appropriate LaTeX environments")
            parts.append("5. Does NOT repeat any section/definition listed in 'ALREADY COVERED'")
        parts.append("")
        parts.append("Output ONLY valid LaTeX content.")

        return "\n".join(parts)

    def _process_slide(self, slide: SlideData) -> bool:
        console.print(
            f"\n[bold cyan]processing slide {slide.slide_number}:[/] {slide.title[:50]}..."
        )

        try:
            query = slide.get_query()
            chunks = self.retriever.retrieve(query, self.retrieval_top_k)
            console.print(f"[dim]retrieved [bold]{len(chunks)}[/] chunks from textbook[/dim]")

            prompt = self._build_prompt(slide, chunks)
            console.print("[dim]generating content...[/dim]")

            generated = self.ollama.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                stream=False,
            )
            if not generated.strip():
                logger.warning("Empty response for slide %s", slide.slide_number)
                return False

            generated = self._clean_latex_output(generated)
            generated = self._remove_duplicate_sections(generated)
            self.chapter_state.extract_from_content(generated)
            self._append_to_output(generated)
            self._update_rolling_context(generated)
            self.progress.mark_completed(slide.slide_number)

            console.print(
                f"[green]✓ slide {slide.slide_number} complete ({len(generated)} chars)[/green]"
            )
            return True
        except Exception as exc:
            logger.error("Failed to process slide %s: %s", slide.slide_number, exc)
            return False

    def run(self) -> None:
        console.print("\n[bold green]===== textbook generation agent =====[/]\n")

        console.print("[bold]step 1:[/] checking ollama connection...")
        if not self.ollama.check_health():
            console.print("[bold red]error: ollama not available[/]")
            console.print(
                f"[dim]ensure ollama is running with model: {self.ollama.model}[/dim]"
            )
            console.print(f"[dim]run: ollama pull {self.ollama.model}[/dim]")
            return
        console.print("[green]✓ ollama connected[/green]")

        console.print("[bold]step 2:[/] initializing output file...")
        self._initialize_output_file()

        slides_to_process = [
            slide
            for slide in self.slides
            if not (self.resume and self.progress.is_completed(slide.slide_number))
        ]
        if not slides_to_process:
            console.print("[yellow]all slides already processed![/yellow]")
            return

        console.print(f"[bold]step 3:[/] processing {len(slides_to_process)} slides...")
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
            task = progress.add_task("generating textbook...", total=len(slides_to_process))
            for slide in slides_to_process:
                if self._process_slide(slide):
                    successful += 1
                else:
                    failed += 1
                progress.update(task, advance=1)

        console.print("\n[bold]step 4:[/] finalizing document...")
        self._finalize_output()

        console.print("\n[bold green]===== generation complete =====[/]\n")
        console.print(f"[bold]slides processed:[/] {successful}/{len(slides_to_process)}")
        if failed > 0:
            console.print(f"[bold red]slides failed:[/] {failed}")
        console.print(f"[bold]output file:[/] {self.output_path}")
        output_size = Path(self.output_path).stat().st_size
        console.print(f"[bold]output size:[/] {output_size:,} bytes")


def create_argparser() -> argparse.ArgumentParser:
    """Create the Ollama CLI parser."""
    parser = argparse.ArgumentParser(
        description="generate latex textbook from slides using rolling context rag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    python main_agent.py
    python main_agent.py --slides lecture.json --output chapter1.tex
    python main_agent.py --model llama3:8b --top-k 10
    python main_agent.py --no-resume
        """,
    )

    parser.add_argument("--slides", "-s", type=str, default="slides.json")
    parser.add_argument("--output", "-o", type=str, default="output.tex")
    parser.add_argument("--db-path", "-d", type=str, default="./chroma_db")
    parser.add_argument("--collection", "-c", type=str, default="textbook_chunks")
    parser.add_argument("--model", "-m", type=str, default="deepseek-r1:32b")
    parser.add_argument("--ollama-url", "-u", type=str, default="http://localhost:11434")
    parser.add_argument("--context-chars", type=int, default=1000)
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for the Ollama backend."""
    parser = create_argparser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if not Path(args.slides).exists():
        console.print(f"[bold red]error:[/] slides json not found: {args.slides}")
        console.print("[dim]run parse_slides.py first to create slides.json[/dim]")
        sys.exit(1)
    if not Path(args.db_path).exists():
        console.print(f"[bold red]error:[/] chromadb not found at: {args.db_path}")
        console.print("[dim]run setup_rag.py first to create the vector store[/dim]")
        sys.exit(1)

    try:
        agent = OllamaTextbookAgent(
            slides_json=args.slides,
            db_path=args.db_path,
            collection_name=args.collection,
            output_path=args.output,
            model=args.model,
            ollama_url=args.ollama_url,
            rolling_context_chars=args.context_chars,
            retrieval_top_k=args.top_k,
            resume=not args.no_resume,
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
