"""Shared utilities for SlideRAG backends."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logger = logging.getLogger("sliderag")


def configure_logging(
    verbose: bool = False,
    extra_loggers: Optional[Sequence[str]] = None,
) -> None:
    """Configure rich logging for CLI entrypoints."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,
    )
    logging.getLogger("sliderag").setLevel(level)
    for name in extra_loggers or []:
        logging.getLogger(name).setLevel(level)


DEFAULT_PREAMBLE = r"""\documentclass[12pt, a4paper]{book}

%-------------------------------------------------------------------------------
% PREAMBLE: PACKAGES, THEMES, AND COMMANDS
%-------------------------------------------------------------------------------

% --- Essential Packages ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}

% --- Remove period after chapter number ---
\makeatletter
\renewcommand{\@chapapp}{Chapter}
\renewcommand{\thechapter}{\arabic{chapter}}
\makeatother

\usepackage{amsmath, amssymb, amsfonts, amsthm}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{float}
\usepackage{mathtools}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{array}
\usepackage{algorithm}
\usepackage{algpseudocode}

% --- Page Layout and Styling ---
\usepackage[
    a4paper,
    left=2.5cm,
    right=2.5cm,
    top=2.5cm,
    bottom=2.5cm,
    headheight=25pt
]{geometry}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage[most]{tcolorbox}
\usepackage{comment}
\usepackage{titlesec}
\usepackage{hyperref}
\usepackage{longtable}

% --- Color Theme Definition ---
\definecolor{PrimaryBlue}{HTML}{0A3D62}
\definecolor{SecondaryGray}{HTML}{F1F3F4}
\definecolor{AccentRed}{HTML}{C0392B}
\definecolor{ContentGray}{HTML}{2C3E50}
\definecolor{SolutionGreen}{HTML}{27AE60}
\definecolor{PropPurple}{HTML}{8E44AD}
\definecolor{HistoryOrange}{HTML}{E67E22}
\definecolor{PhilosophyTeal}{HTML}{16A085}

% --- Hyperref Setup ---
\hypersetup{
    colorlinks=true,
    linkcolor=PrimaryBlue,
    urlcolor=PrimaryBlue,
    citecolor=PrimaryBlue,
    pdftitle={AI: A Deep Dive},
    pdfauthor={SlideRAG Agent},
    pdfsubject={Artificial Intelligence}
}

% --- Typography and Document Settings ---
\renewcommand{\familydefault}{\sfdefault}
\color{ContentGray}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\linespread{1.15}

% --- Section Spacing ---
\titlespacing*{\section}
{0pt}{1.5ex plus 0.5ex minus 0.2ex}{1ex plus 0.2ex}
\titlespacing*{\subsection}
{0pt}{1.2ex plus 0.4ex minus 0.2ex}{0.8ex plus 0.2ex}

% --- Header and Footer Configuration ---
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\textcolor{PrimaryBlue}{\sffamily\bfseries\thepage}}
\fancyhead[LO]{\textcolor{PrimaryBlue}{\sffamily\rightmark}}
\fancyhead[RE]{\textcolor{PrimaryBlue}{\sffamily\leftmark}}
\renewcommand{\headrulewidth}{0.6pt}
\renewcommand{\footrulewidth}{0pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{PrimaryBlue}\leaders\hrule height \headrulewidth\hfill}}

% --- Custom Theorem and Definition Environments ---
\tcbuselibrary{theorems, skins, breakable}

\makeatletter
\providecommand\tcb@titleafterhead{}
\pgfkeys{/tcb/title after head/.code={%
  \def\tcb@titleafterhead{#1}%
  \tcbset{title code/.add code={}{\hspace{0.25em}\tcb@titleafterhead}}%
}}
\makeatother

% Definition
\newtcbtheorem[auto counter, number within=section]{definition}{Definition}{
    enhanced,
    breakable,
    colback=SecondaryGray,
    colframe=PrimaryBlue,
    coltitle=white,
    colbacktitle=PrimaryBlue,
    fonttitle=\bfseries,
    attach boxed title to top left={yshift=-0.25mm-\tcboxedtitleheight/2, xshift=5mm},
    boxed title style={
        arc=3mm,
        outer arc=3mm,
        boxrule=0.5mm,
    },
    before skip=10pt plus 2pt,
    after skip=10pt plus 2pt
}{def}

% Theorem
\newtcbtheorem[auto counter, number within=section]{theorem}{Theorem}{
    enhanced,
    breakable,
    colback=white,
    colframe=AccentRed!75!black,
    coltitle=white,
    colbacktitle=AccentRed!75!black,
    fonttitle=\bfseries,
    attach boxed title to top left={yshift=-0.25mm-\tcboxedtitleheight/2, xshift=5mm},
    boxed title style={
        arc=3mm,
        outer arc=3mm,
        boxrule=0.5mm,
    },
    before skip=10pt plus 2pt,
    after skip=10pt plus 2pt
}{thm}

% Proposition
\newtcbtheorem[auto counter, number within=section]{proposition}{Proposition}{
    enhanced,
    breakable,
    colback=PropPurple!5!white,
    colframe=PropPurple!75!black,
    coltitle=white,
    colbacktitle=PropPurple!75!black,
    fonttitle=\bfseries,
    attach boxed title to top left={yshift=-0.25mm-\tcboxedtitleheight/2, xshift=5mm},
    boxed title style={
        arc=3mm,
        outer arc=3mm,
        boxrule=0.5mm,
    },
    before skip=10pt plus 2pt,
    after skip=10pt plus 2pt
}{prop}

% Key Concept Box
\newtcolorbox{keyconcept}[1][]{
    enhanced,
    breakable,
    colback=PrimaryBlue!5!white,
    colframe=PrimaryBlue,
    fonttitle=\bfseries\color{white},
    title={Key Concept},
    attach boxed title to top center={yshift=-2mm},
    boxed title style={colback=PrimaryBlue},
    #1
}

% Historical Note Box
\newtcolorbox{historicalnote}[1][]{
    enhanced,
    breakable,
    colback=HistoryOrange!5!white,
    colframe=HistoryOrange,
    fonttitle=\bfseries\color{white},
    title={Historical Note},
    attach boxed title to top left={yshift=-2mm, xshift=5mm},
    boxed title style={colback=HistoryOrange},
    #1
}

% Philosophy Box
\newtcolorbox{philosophybox}[1][]{
    enhanced,
    breakable,
    colback=PhilosophyTeal!5!white,
    colframe=PhilosophyTeal,
    fonttitle=\bfseries\color{white},
    title={Philosophical Perspective},
    attach boxed title to top left={yshift=-2mm, xshift=5mm},
    boxed title style={colback=PhilosophyTeal},
    #1
}

% Example
\newtcbtheorem[auto counter, number within=section]{example}{Example}{
    breakable,
    colback=white,
    colframe=PrimaryBlue!20!white,
    fonttitle=\bfseries\color{PrimaryBlue},
    title after head={:},
    arc=0mm,
    before skip=10pt plus 2pt,
    after skip=10pt plus 2pt
}{ex}

% Remark
\newtcbtheorem[auto counter, number within=section]{remark}{Remark}{
    breakable,
    colback=SecondaryGray!50!white,
    colframe=PrimaryBlue!50!white,
    fonttitle=\bfseries\color{PrimaryBlue},
    title after head={:},
    arc=0mm,
    borderline west={2mm}{0pt}{PrimaryBlue!60!white},
    before skip=10pt plus 2pt,
    after skip=10pt plus 2pt
}{rem}

% Important Box
\newtcolorbox{importantbox}[1][]{
    enhanced,
    breakable,
    colback=AccentRed!5!white,
    colframe=AccentRed,
    fonttitle=\bfseries\color{white},
    title={Important},
    attach boxed title to top left={yshift=-2mm, xshift=5mm},
    boxed title style={colback=AccentRed},
    #1
}

% Proof environment styling
\tcolorboxenvironment{proof}{
    blanker,
    breakable,
    left=5mm,
    borderline west={1mm}{0pt}{SolutionGreen!70!black},
    before skip=10pt plus 2pt,
    after skip=10pt plus 2pt
}

% --- Mathematical Notation Shortcuts ---
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Cov}{\operatorname{Cov}}
\newcommand{\Prob}{\mathbb{P}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\iid}{\stackrel{\text{i.i.d.}}{\sim}}
\newcommand{\convp}{\xrightarrow{p}}
\newcommand{\convd}{\xrightarrow{d}}

% --- Document Info ---
\title{\textcolor{PrimaryBlue}{\textbf{AI: A Deep Dive}} \\ \Large Comprehensive Textbook \\ \vspace{0.5cm} \large Generated by SlideRAG Agent}
\author{Generated from Course Slides and Textbook \\ Using Rolling Context RAG}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""


@dataclass
class SlideData:
    """Represents a parsed slide with its content."""

    slide_number: int
    title: str
    content: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlideData":
        return cls(
            slide_number=data.get("slide_number", 0),
            title=data.get("title", ""),
            content=data.get("content", ""),
        )

    def get_query(self) -> str:
        """Generate a retrieval query from the slide title and body."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.content:
            parts.append(self.content[:200])
        return " ".join(parts)


class ChromaRetriever:
    """Retrieve relevant chunks from ChromaDB."""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "textbook_chunks",
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        if not self.db_path.exists():
            raise FileNotFoundError(f"ChromaDB not found at: {db_path}")

        logger.info("Connecting to ChromaDB at: %s", db_path)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(collection_name)
        logger.info(
            "Connected to collection '%s' with %s documents",
            collection_name,
            self.collection.count(),
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve top-k chunks for a single query."""
        if not query.strip():
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count()),
            )
            return results.get("documents", [[]])[0]
        except Exception as exc:
            logger.warning("Retrieval failed: %s", exc)
            return []

    def retrieve_for_batch(
        self,
        queries: Iterable[str],
        top_k_per_query: int = 4,
    ) -> List[str]:
        """Retrieve and deduplicate chunks for multiple slide queries."""
        all_chunks: List[str] = []
        seen_hashes = set()

        for query in queries:
            for chunk in self.retrieve(query, top_k_per_query):
                chunk_hash = hash(chunk[:150])
                if chunk_hash in seen_hashes:
                    continue
                seen_hashes.add(chunk_hash)
                all_chunks.append(chunk)

        return all_chunks

    def get_document_count(self) -> int:
        """Return the number of chunks stored in the collection."""
        return self.collection.count()


class ProgressTracker:
    """Track completed slide numbers for resume support."""

    def __init__(self, output_path: str):
        self.progress_file = Path(output_path).with_suffix(".progress.json")
        self.completed_slides: set[int] = set()
        self._load_progress()

    def _load_progress(self) -> None:
        if not self.progress_file.exists():
            return
        try:
            with open(self.progress_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.completed_slides = set(data.get("completed_slides", []))
        except Exception as exc:
            logger.warning("Failed to load progress: %s", exc)

    def _save_progress(self) -> None:
        try:
            with open(self.progress_file, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "completed_slides": sorted(self.completed_slides),
                        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    handle,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Failed to save progress: %s", exc)

    def mark_completed(self, slide_number: int) -> None:
        self.completed_slides.add(slide_number)
        self._save_progress()

    def mark_batch_completed(self, slide_numbers: Iterable[int]) -> None:
        self.completed_slides.update(slide_numbers)
        self._save_progress()

    def is_completed(self, slide_number: int) -> bool:
        return slide_number in self.completed_slides

    def is_batch_completed(self, slide_numbers: Iterable[int]) -> bool:
        return all(slide_number in self.completed_slides for slide_number in slide_numbers)

    def reset(self) -> None:
        self.completed_slides = set()
        if self.progress_file.exists():
            self.progress_file.unlink()


class ChapterStateTracker:
    """Track generated structure to reduce repetition across prompts."""

    def __init__(self):
        self.sections: List[str] = []
        self.subsections: List[str] = []
        self.topics_covered: set[str] = set()
        self.definitions_given: set[str] = set()
        self.examples_given: set[str] = set()
        self.figure_labels: set[str] = set()

    def extract_from_content(self, content: str) -> None:
        self.sections.extend(re.findall(r"\\section\{([^}]+)\}", content))
        self.subsections.extend(re.findall(r"\\subsection\{([^}]+)\}", content))
        self.definitions_given.update(
            re.findall(r"\\begin\{definition\}[^{]*\{([^}]+)\}", content)
        )
        self.examples_given.update(
            re.findall(r"\\begin\{example\}[^{]*\{([^}]+)\}", content)
        )
        self.figure_labels.update(re.findall(r"\\label\{(fig:[^}]+)\}", content))

        topic_matches = re.findall(
            r"\\begin\{keyconcept\}.*?\\end\{keyconcept\}",
            content,
            re.DOTALL,
        )
        for topic in topic_matches:
            words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", topic)
            self.topics_covered.update(words[:3])

    def get_context_summary(self) -> str:
        parts = []
        if self.sections:
            parts.append(f"SECTIONS ALREADY WRITTEN: {', '.join(self.sections[-8:])}")
        if self.subsections:
            parts.append(
                f"SUBSECTIONS ALREADY WRITTEN: {', '.join(self.subsections[-12:])}"
            )
        if self.definitions_given:
            parts.append(
                "DEFINITIONS ALREADY PROVIDED: "
                f"{', '.join(list(self.definitions_given)[-10:])}"
            )
        if self.examples_given:
            parts.append(
                f"EXAMPLES ALREADY GIVEN: {', '.join(list(self.examples_given)[-8:])}"
            )
        return "\n".join(parts)

    def is_duplicate_section(self, title: str) -> bool:
        title_lower = title.lower().strip()
        for existing in self.sections:
            existing_lower = existing.lower().strip()
            if title_lower == existing_lower:
                return True
            if title_lower.startswith(existing_lower[:20]) or existing_lower.startswith(
                title_lower[:20]
            ):
                if len(title_lower) > 10 and len(existing_lower) > 10:
                    return True
        return False


class BaseTextbookAgent:
    """Shared state and LaTeX output helpers for backend-specific agents."""

    def __init__(
        self,
        slides_json: str,
        db_path: str = "./chroma_db",
        collection_name: str = "textbook_chunks",
        output_path: str = "output.tex",
        rolling_context_chars: int = 2500,
        retrieval_top_k: int = 5,
        resume: bool = True,
    ):
        self.slides_json = slides_json
        self.output_path = output_path
        self.rolling_context_chars = rolling_context_chars
        self.retrieval_top_k = retrieval_top_k
        self.resume = resume

        self.slides: List[SlideData] = self._load_slides()
        self.retriever = ChromaRetriever(db_path, collection_name)
        self.progress = ProgressTracker(output_path)
        self.chapter_state = ChapterStateTracker()
        self.rolling_context = ""

    def _load_slides(self) -> List[SlideData]:
        slides_path = Path(self.slides_json)
        if not slides_path.exists():
            raise FileNotFoundError(f"Slides JSON not found: {self.slides_json}")

        with open(slides_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        slides = [SlideData.from_dict(item) for item in data]
        logger.info("Loaded %s slides from %s", len(slides), self.slides_json)
        return slides

    def _get_rolling_context(self) -> str:
        if not self.rolling_context:
            return ""
        return self.rolling_context[-self.rolling_context_chars :]

    def _update_rolling_context(self, new_content: str) -> None:
        self.rolling_context += new_content
        max_buffer = self.rolling_context_chars * 3
        if len(self.rolling_context) > max_buffer:
            self.rolling_context = self.rolling_context[-self.rolling_context_chars :]

    def _is_title_or_outline_slide(self, slide: SlideData) -> bool:
        content = slide.content.strip()
        if len(content) < 50:
            return True

        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if len(lines) <= 2:
            return True

        outline_keywords = ["outline", "overview", "agenda", "contents", "topics"]
        return any(keyword in slide.title.lower() for keyword in outline_keywords)

    def _initialize_output_file(self) -> None:
        output_path = Path(self.output_path)

        if self.resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as handle:
                content = handle.read()

            doc_start = content.find(r"\begin{document}")
            if doc_start != -1:
                self.rolling_context = content[
                    doc_start + len(r"\begin{document}") :
                ].replace(r"\end{document}", "")
            self.chapter_state.extract_from_content(content)
            logger.info(
                "Resuming from %s with %s chars of rolling context",
                output_path,
                len(self.rolling_context),
            )
            return

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(DEFAULT_PREAMBLE)
        logger.info("Created new output file with preamble: %s", output_path)

    def _append_to_output(self, content: str) -> None:
        with open(self.output_path, "a", encoding="utf-8") as handle:
            handle.write(content)
            handle.write("\n\n")

    def _finalize_output(self) -> None:
        output_path = Path(self.output_path)
        if not output_path.exists():
            return

        with open(output_path, "r+", encoding="utf-8") as handle:
            content = handle.read()
            if content.rstrip().endswith(r"\end{document}"):
                return
            if content and not content.endswith("\n"):
                handle.write("\n")
            handle.write(r"\end{document}" + "\n")

    def _clean_latex_output(self, text: str) -> str:
        code_blocks = re.findall(r"```(?:latex|tex)?\s*\n?(.*?)```", text, re.DOTALL)
        if code_blocks:
            text = "\n\n".join(code_blocks)

        text = re.sub(r"```(?:latex|tex)?\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        meta_patterns = [
            r"^Here is the .*?(?:LaTeX|content|chapter|section).*?:\s*",
            r"^Below is the .*?(?:LaTeX|content|chapter|section).*?:\s*",
            r"^The following .*?(?:LaTeX|content|chapter|section).*?:\s*",
            r"This (?:LaTeX )?content provides .*?(?:guidelines|requirements|specifications)\.?\s*$",
            r"This concludes .*?(?:chapter|section|content)\.?\s*$",
            r"^I will now .*?:\s*",
            r"^Let me .*?:\s*",
            r"^Sure[,!]? .*?:\s*",
            r"^Okay[,!]? .*?:\s*",
            r"^Here you go.*?:\s*",
            r"Show thinking\s*Gemini said",
            r"^Show thinking",
            r"^Gemini said",
        ]
        for pattern in meta_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

        for env in ["keyconcept", "historicalnote", "philosophybox", "importantbox"]:
            text = re.sub(
                r"\\begin\{" + env + r"\}\[([^=\]]+)\]",
                r"\\begin{" + env + r"}[title={\1}]",
                text,
            )

        for env in ["definition", "theorem", "proposition", "example", "remark"]:
            text = re.sub(
                r"\\begin\{" + env + r"\}\[([^\]]+)\]",
                r"\\begin{" + env + r"}{\1}{}",
                text,
            )
            text = re.sub(
                r"\\begin\{" + env + r"\}(?!\{)",
                r"\\begin{" + env + r"}{}{}",
                text,
            )

        unicode_replacements = {
            "\u200b": "",
            "′": "'",
            "∀": r"$\forall$ ",
            "∈": r" $\in$ ",
            "∗": r"$*$",
            "": r"$\not$",
            "⋃": r"$\bigcup$",
            "⋂": r"$\bigcap$",
            "∣": r"$\mid$",
            "−": r"$-$",
            "∞": r"$\infty$",
            "Δ": r"$\Delta$",
            "⋅": r"$\cdot$",
            "α": r"$\alpha$",
            "β": r"$\beta$",
            "≈": r"$\approx$",
            "≤": r"$\leq$",
            "≥": r"$\geq$",
            "≠": r"$\neq$",
        }
        for source, replacement in unicode_replacements.items():
            text = text.replace(source, replacement)

        text = re.sub(r"\\documentclass.*?\n", "", text)
        text = re.sub(r"\\begin\{document\}", "", text)
        text = re.sub(r"\\end\{document\}", "", text)
        text = re.sub(r"\\usepackage.*?\n", "", text)
        text = re.sub(r"\\maketitle", "", text)
        text = re.sub(r"\\tableofcontents", "", text)

        text = re.sub(r"\[?Source\s*\d+\]?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\(see Source\s*\d+\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"as discussed in Source\s*\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"according to Source\s*\d+", "", text, flags=re.IGNORECASE)

        text = re.sub(r"Figure~?\\ref\{fig:[^}]+\}", "", text)
        text = re.sub(r"\(see Figure~?\\ref\{[^}]+\}\)", "", text)
        text = re.sub(r"discussed in Section\s*\d+(\.\d+)?", "discussed earlier", text)

        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"  +", " ", text)
        return text.strip()

    def _remove_duplicate_sections(self, text: str) -> str:
        for title in re.findall(r"\\section\{([^}]+)\}", text):
            if self.chapter_state.is_duplicate_section(title):
                logger.warning("Potential duplicate section detected: %s", title)

        for title in re.findall(r"\\subsection\{([^}]+)\}", text):
            title_lower = title.lower()
            if any(title_lower == existing.lower() for existing in self.chapter_state.subsections):
                logger.warning("Potential duplicate subsection detected: %s", title)

        return text
