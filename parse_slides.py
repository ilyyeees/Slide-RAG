#!/usr/bin/env python3
"""
parse_slides.py - convert slide pdf to structured json

this module handles:
1. loading and parsing slides.pdf
2. extracting slide titles and content
3. preserving bullet point structure
4. outputting structured json for the generation pipeline

usage:
    python parse_slides.py [--pdf slides.pdf] [--output slides.json]

author: sliderag project
license: mit
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # pymupdf
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# try to import pdfplumber as fallback
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# configure rich console
console = Console()

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("parse_slides")


class TextBlock:
    """
    represents a block of text extracted from a pdf page.
    
    attributes:
        text: the text content
        bbox: bounding box (x0, y0, x1, y1)
        font_size: estimated font size
        is_bold: whether the text appears bold
        line_number: vertical position ranking on the page
    """
    
    def __init__(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        font_size: float = 12.0,
        is_bold: bool = False,
        font_name: str = ""
    ):
        self.text = text.strip()
        self.bbox = bbox
        self.font_size = font_size
        self.is_bold = is_bold
        self.font_name = font_name
        
        # calculate center y position for sorting
        self.center_y = (bbox[1] + bbox[3]) / 2
        self.center_x = (bbox[0] + bbox[2]) / 2
    
    @property
    def x0(self) -> float:
        return self.bbox[0]
    
    @property
    def y0(self) -> float:
        return self.bbox[1]
    
    @property
    def x1(self) -> float:
        return self.bbox[2]
    
    @property
    def y1(self) -> float:
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    def __repr__(self) -> str:
        return f"TextBlock('{self.text[:30]}...', size={self.font_size:.1f})"


class SlideExtractor:
    """
    extract structured content from presentation slides using pymupdf.
    
    this class handles the complexities of slide extraction:
    - identifying slide titles (typically largest/top text)
    - extracting bullet points and preserving hierarchy
    - handling various slide layouts
    - cleaning and normalizing text
    
    attributes:
        pdf_path: path to the slide pdf
        doc: the opened pymupdf document
    """
    
    def __init__(self, pdf_path: str):
        """
        initialize the slide extractor.
        
        args:
            pdf_path: path to the slide pdf file
            
        raises:
            filenotfounderror: if pdf does not exist
            exception: if pdf cannot be opened
        """
        self.pdf_path = Path(pdf_path)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"slide pdf not found: {pdf_path}")
        
        if not self.pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"file is not a pdf: {pdf_path}")
        
        try:
            self.doc = fitz.open(str(self.pdf_path))
            logger.info(f"opened slide pdf: {pdf_path} ({len(self.doc)} slides)")
        except Exception as e:
            logger.error(f"failed to open pdf: {e}")
            raise
    
    def _extract_text_blocks(self, page: fitz.Page) -> List[TextBlock]:
        """
        extract text blocks from a page with font information.
        
        uses pymupdf's text extraction with detailed block information
        to get font sizes and positions for each text element.
        
        args:
            page: pymupdf page object
            
        returns:
            list of textblock objects sorted by vertical position
        """
        blocks = []
        
        # get detailed text blocks with font info
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            
            for line in block.get("lines", []):
                line_text = ""
                line_font_size = 0
                line_is_bold = False
                line_font_name = ""
                
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if not span_text:
                        continue
                    
                    line_text += span_text + " "
                    
                    # track the largest font in the line
                    span_size = span.get("size", 12)
                    if span_size > line_font_size:
                        line_font_size = span_size
                        line_font_name = span.get("font", "")
                        line_is_bold = "bold" in line_font_name.lower() or "black" in line_font_name.lower()
                
                line_text = line_text.strip()
                if line_text:
                    bbox = line.get("bbox", (0, 0, 0, 0))
                    blocks.append(TextBlock(
                        text=line_text,
                        bbox=bbox,
                        font_size=line_font_size,
                        is_bold=line_is_bold,
                        font_name=line_font_name
                    ))
        
        # sort blocks by vertical position (top to bottom)
        blocks.sort(key=lambda b: (b.y0, b.x0))
        
        return blocks
    
    def _identify_title(self, blocks: List[TextBlock], page_height: float) -> Tuple[Optional[str], List[TextBlock]]:
        """
        identify the slide title from text blocks.
        
        title identification heuristics:
        1. usually in the top 25% of the slide
        2. typically has larger font size than body text
        3. often bold
        4. usually a single line (not too long)
        
        args:
            blocks: list of text blocks sorted by position
            page_height: height of the page for position calculations
            
        returns:
            tuple of (title string or none, remaining blocks)
        """
        if not blocks:
            return None, []
        
        # find blocks in top portion of slide
        top_threshold = page_height * 0.25
        top_blocks = [b for b in blocks if b.y0 < top_threshold]
        
        if not top_blocks:
            # no blocks in top area, check first block
            top_blocks = blocks[:1]
        
        # find the block with largest font in top area
        if top_blocks:
            title_block = max(top_blocks, key=lambda b: b.font_size)
            
            # verify it looks like a title
            # - not too long (typically < 100 chars)
            # - larger than average font or at very top
            if len(title_block.text) < 150:
                remaining = [b for b in blocks if b != title_block]
                return title_block.text, remaining
        
        # fallback: use first block if it looks like a title
        first_block = blocks[0]
        if len(first_block.text) < 150 and first_block.y0 < page_height * 0.3:
            return first_block.text, blocks[1:]
        
        return None, blocks
    
    def _format_content(self, blocks: List[TextBlock]) -> str:
        """
        format remaining text blocks as slide content.
        
        handles:
        - bullet points (detected by leading characters like •, -, *, etc.)
        - indentation levels (based on x position)
        - paragraph grouping
        
        args:
            blocks: list of text blocks (excluding title)
            
        returns:
            formatted content string with bullet structure preserved
        """
        if not blocks:
            return ""
        
        # bullet point markers
        bullet_markers = ['•', '●', '○', '▪', '▫', '◦', '-', '*', '→', '➢', '➤', '►', '‣']
        
        # find minimum x position for indentation reference
        min_x = min(b.x0 for b in blocks)
        
        lines = []
        
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            
            # check if line starts with bullet
            has_bullet = any(text.startswith(m) for m in bullet_markers)
            
            # calculate indentation level
            indent_level = 0
            if block.x0 > min_x + 20:  # significant indentation threshold
                indent_level = int((block.x0 - min_x) / 30)  # ~30 pixels per level
            
            # format the line
            if has_bullet:
                # standardize bullet to "-"
                for marker in bullet_markers:
                    if text.startswith(marker):
                        text = text[len(marker):].strip()
                        break
                prefix = "  " * indent_level + "- "
            else:
                prefix = "  " * indent_level
            
            lines.append(prefix + text)
        
        return "\n".join(lines)
    
    def _clean_slide_content(self, title: Optional[str], content: str) -> Tuple[str, str]:
        """
        clean and normalize slide content.
        
        handles:
        - removing page numbers
        - normalizing whitespace
        - removing header/footer remnants
        - trimming excessive blank lines
        
        args:
            title: the slide title
            content: the slide content
            
        returns:
            tuple of (cleaned title, cleaned content)
        """
        # clean title
        if title:
            # remove leading numbers like "1." or "Chapter 1:"
            title = re.sub(r'^\d+[\.\)]\s*', '', title)
            title = re.sub(r'^chapter\s+\d+[:\s]*', '', title, flags=re.IGNORECASE)
            title = title.strip()
        
        # clean content
        if content:
            # remove standalone numbers (likely page numbers)
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                # skip lines that are just numbers
                if stripped and stripped.isdigit():
                    continue
                # skip very short lines that look like headers/footers
                if len(stripped) < 3 and not stripped.startswith('-'):
                    continue
                cleaned_lines.append(line)
            content = '\n'.join(cleaned_lines)
            
            # normalize multiple blank lines
            content = re.sub(r'\n{3,}', '\n\n', content)
        
        return title or "", content.strip()
    
    def extract_slide(self, page_num: int) -> Dict[str, Any]:
        """
        extract structured content from a single slide.
        
        args:
            page_num: zero-indexed page number
            
        returns:
            dictionary with slide_number, title, and content
        """
        page = self.doc[page_num]
        page_height = page.rect.height
        
        # extract text blocks
        blocks = self._extract_text_blocks(page)
        
        if not blocks:
            # empty slide
            return {
                "slide_number": page_num + 1,
                "title": "",
                "content": "",
                "is_empty": True
            }
        
        # identify title
        title, content_blocks = self._identify_title(blocks, page_height)
        
        # format content
        content = self._format_content(content_blocks)
        
        # clean everything
        title, content = self._clean_slide_content(title, content)
        
        return {
            "slide_number": page_num + 1,
            "title": title,
            "content": content,
            "is_empty": not title and not content
        }
    
    def extract_all_slides(self, skip_empty: bool = True) -> List[Dict[str, Any]]:
        """
        extract all slides from the pdf.
        
        args:
            skip_empty: if true, exclude slides with no content
            
        returns:
            list of slide dictionaries
        """
        slides = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("parsing slides...", total=len(self.doc))
            
            for page_num in range(len(self.doc)):
                try:
                    slide = self.extract_slide(page_num)
                    
                    if skip_empty and slide.get("is_empty", False):
                        logger.debug(f"skipping empty slide {page_num + 1}")
                    else:
                        # remove internal flag before adding
                        slide.pop("is_empty", None)
                        slides.append(slide)
                        
                except Exception as e:
                    logger.warning(f"failed to parse slide {page_num + 1}: {e}")
                
                progress.update(task, advance=1)
        
        logger.info(f"extracted {len(slides)} slides from {len(self.doc)} pages")
        return slides
    
    def close(self):
        """close the pdf document."""
        if self.doc:
            self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PdfPlumberExtractor:
    """
    fallback slide extractor using pdfplumber.
    
    used when pymupdf extraction produces poor results,
    particularly for slides with complex layouts or unusual fonts.
    
    this is a simpler extractor that uses pdfplumber's
    more robust text extraction for difficult pdfs.
    """
    
    def __init__(self, pdf_path: str):
        """
        initialize the pdfplumber extractor.
        
        args:
            pdf_path: path to the slide pdf
        """
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber not installed. install with: pip install pdfplumber")
        
        self.pdf_path = Path(pdf_path)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"pdf not found: {pdf_path}")
        
        self.pdf = pdfplumber.open(str(self.pdf_path))
        logger.info(f"opened pdf with pdfplumber: {pdf_path} ({len(self.pdf.pages)} pages)")
    
    def extract_all_slides(self, skip_empty: bool = True) -> List[Dict[str, Any]]:
        """
        extract all slides using pdfplumber.
        
        uses simpler heuristics than the pymupdf extractor:
        - first line is title
        - remaining lines are content
        
        args:
            skip_empty: if true, skip empty slides
            
        returns:
            list of slide dictionaries
        """
        slides = []
        
        for page_num, page in enumerate(self.pdf.pages):
            try:
                text = page.extract_text() or ""
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                
                if not lines:
                    if not skip_empty:
                        slides.append({
                            "slide_number": page_num + 1,
                            "title": "",
                            "content": ""
                        })
                    continue
                
                # first non-empty line is title
                title = lines[0] if lines else ""
                content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                
                slides.append({
                    "slide_number": page_num + 1,
                    "title": title,
                    "content": content
                })
                
            except Exception as e:
                logger.warning(f"pdfplumber failed on page {page_num + 1}: {e}")
        
        return slides
    
    def close(self):
        """close the pdf."""
        if self.pdf:
            self.pdf.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SlideParser:
    """
    main orchestrator for slide parsing pipeline.
    
    coordinates:
    1. pdf extraction (pymupdf primary, pdfplumber fallback)
    2. content structuring
    3. json output
    
    usage:
        parser = SlideParser("slides.pdf", "slides.json")
        slides = parser.run()
    """
    
    def __init__(
        self,
        pdf_path: str,
        output_path: str = "slides.json",
        use_fallback: bool = False,
        skip_empty: bool = True
    ):
        """
        initialize the slide parser.
        
        args:
            pdf_path: path to the slide pdf
            output_path: path for json output
            use_fallback: if true, use pdfplumber instead of pymupdf
            skip_empty: if true, skip slides with no content
        """
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.use_fallback = use_fallback
        self.skip_empty = skip_empty
    
    def run(self) -> List[Dict[str, Any]]:
        """
        run the slide parsing pipeline.
        
        returns:
            list of parsed slide dictionaries
        """
        console.print("\n[bold green]===== slide parsing pipeline =====[/]\n")
        
        # choose extractor
        if self.use_fallback and PDFPLUMBER_AVAILABLE:
            console.print("[bold]using pdfplumber extractor[/]")
            extractor = PdfPlumberExtractor(self.pdf_path)
        else:
            console.print("[bold]using pymupdf extractor[/]")
            extractor = SlideExtractor(self.pdf_path)
        
        try:
            # extract slides
            console.print("[bold]step 1:[/] extracting slide content...")
            slides = extractor.extract_all_slides(skip_empty=self.skip_empty)
            
            if not slides:
                console.print("[bold yellow]warning: no slides extracted[/]")
                return []
            
            # save to json
            console.print("[bold]step 2:[/] saving to json...")
            self._save_to_json(slides)
            
            # print summary
            self._print_summary(slides)
            
            console.print(f"\n[bold green]✓ parsing complete![/]")
            console.print(f"[dim]output saved to: {self.output_path}[/]\n")
            
            return slides
            
        finally:
            extractor.close()
    
    def _save_to_json(self, slides: List[Dict[str, Any]]) -> None:
        """save slides to json file."""
        output_path = Path(self.output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(slides, f, indent=2, ensure_ascii=False)
        
        logger.info(f"saved {len(slides)} slides to {output_path}")
    
    def _print_summary(self, slides: List[Dict[str, Any]]) -> None:
        """print a summary table of parsed slides."""
        table = Table(title="parsed slides summary")
        table.add_column("slide", style="cyan", justify="right")
        table.add_column("title", style="green", max_width=50)
        table.add_column("content length", style="yellow", justify="right")
        
        for slide in slides[:10]:  # show first 10
            title = slide.get("title", "")[:47] + "..." if len(slide.get("title", "")) > 50 else slide.get("title", "")
            table.add_row(
                str(slide["slide_number"]),
                title or "[no title]",
                str(len(slide.get("content", "")))
            )
        
        if len(slides) > 10:
            table.add_row("...", f"[dim]({len(slides) - 10} more slides)[/dim]", "...")
        
        console.print(table)
        console.print(f"\n[bold]total slides:[/] {len(slides)}")


def create_argparser() -> argparse.ArgumentParser:
    """create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="parse slide pdf into structured json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    python parse_slides.py
    python parse_slides.py --pdf lecture.pdf --output lecture.json
    python parse_slides.py --fallback  # use pdfplumber instead of pymupdf
        """
    )
    
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        default="slides.pdf",
        help="path to the slide pdf file (default: slides.pdf)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="slides.json",
        help="output json file path (default: slides.json)"
    )
    
    parser.add_argument(
        "--fallback", "-f",
        action="store_true",
        help="use pdfplumber instead of pymupdf for extraction"
    )
    
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="include slides with no content"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="enable verbose debug logging"
    )
    
    return parser


def main():
    """main entry point for slide parsing."""
    parser = create_argparser()
    args = parser.parse_args()
    
    # set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # validate pdf exists
    if not Path(args.pdf).exists():
        console.print(f"[bold red]error:[/] pdf file not found: {args.pdf}")
        console.print("[dim]make sure slides.pdf exists in the current directory[/]")
        sys.exit(1)
    
    try:
        # run the parsing pipeline
        slide_parser = SlideParser(
            pdf_path=args.pdf,
            output_path=args.output,
            use_fallback=args.fallback,
            skip_empty=not args.include_empty
        )
        
        slides = slide_parser.run()
        
        # print final stats
        console.print(f"[bold]slides extracted:[/] {len(slides)}")
        console.print(f"[bold]output file:[/] {args.output}")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]file error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]error:[/] {e}")
        logger.exception("slide parsing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
