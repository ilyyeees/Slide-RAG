#!/usr/bin/env python3
"""
main_agent.py - rolling context rag textbook generation agent

this module implements the core generation loop:
1. load parsed slides from json
2. retrieve relevant textbook chunks from chromadb
3. build rolling context prompts
4. call ollama api for latex generation
5. stream output to tex file

usage:
    python main_agent.py [--slides slides.json] [--output output.tex] [--model deepseek-r1:8b]

author: sliderag project
license: mit
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import chromadb
from chromadb.config import Settings
import httpx
import requests
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.markdown import Markdown

# configure rich console
console = Console()

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("main_agent")


# default latex preamble for output file
# this is the preamble from the instructions
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
    before skip=10pt,
    after skip=10pt,
    borderline west={1mm}{0pt}{SolutionGreen}
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

% ============================================
% DOCUMENT INFO
% ============================================
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
    """represents a parsed slide with its content."""
    slide_number: int
    title: str
    content: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlideData":
        return cls(
            slide_number=data.get("slide_number", 0),
            title=data.get("title", ""),
            content=data.get("content", "")
        )
    
    def get_query(self) -> str:
        """generate a search query from slide content."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.content:
            # take first 200 chars of content for query
            parts.append(self.content[:200])
        return " ".join(parts)


class OllamaClient:
    """
    http client for ollama api.
    
    handles:
    - connection to local ollama server
    - model generation requests
    - streaming responses
    - error handling and retries
    
    attributes:
        base_url: ollama api base url
        model: model name to use
        timeout: request timeout in seconds
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1:32b",
        timeout: int = 300
    ):
        """
        initialize the ollama client.
        
        args:
            base_url: ollama api base url
            model: model name to use for generation
            timeout: request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        
        logger.info(f"initialized ollama client: {base_url}, model: {model}")
    
    def check_health(self) -> bool:
        """
        check if ollama server is running and model is available.
        
        returns:
            true if server is healthy and model exists
        """
        try:
            # check server is up
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                logger.error("ollama server not responding")
                return False
            
            # check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # check exact match or partial match (model:tag format)
            model_base = self.model.split(":")[0]
            available = any(
                self.model in name or model_base in name
                for name in model_names
            )
            
            if not available:
                logger.warning(f"model '{self.model}' not found. available: {model_names}")
                return False
            
            logger.info(f"ollama health check passed, model '{self.model}' available")
            return True
            
        except requests.RequestException as e:
            logger.error(f"ollama health check failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        max_retries: int = 3
    ) -> str:
        """
        generate text using ollama.
        
        args:
            prompt: the user prompt
            system: optional system prompt
            stream: if true, stream the response
            max_retries: number of retry attempts on failure
            
        returns:
            generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 8192,  # increased for rtx 5090
            }
        }
        
        if system:
            payload["system"] = system
        
        for attempt in range(max_retries):
            try:
                if stream:
                    return self._generate_streaming(payload)
                else:
                    return self._generate_sync(payload)
                    
            except (requests.RequestException, httpx.HTTPError) as e:
                logger.warning(f"generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    raise
        
        return ""
    
    def _generate_sync(self, payload: Dict[str, Any]) -> str:
        """synchronous generation request."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _generate_streaming(self, payload: Dict[str, Any]) -> str:
        """streaming generation request."""
        payload["stream"] = True
        
        full_response = []
        
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            full_response.append(chunk)
                            
                            # check if done
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        
        return "".join(full_response)
    
    def generate_streaming_chunks(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        generate text with streaming, yielding chunks.
        
        useful for real-time output to file.
        
        args:
            prompt: the user prompt
            system: optional system prompt
            
        yields:
            text chunks as they are generated
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 8192,  # increased for rtx 5090
            }
        }
        
        if system:
            payload["system"] = system
        
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                            
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue


class ChromaRetriever:
    """
    retrieve relevant chunks from chromadb.
    
    handles:
    - connecting to existing chromadb instance
    - semantic similarity search
    - returning relevant context for generation
    
    attributes:
        db_path: path to chromadb persistence directory
        collection_name: name of the collection to query
        collection: the chromadb collection object
    """
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "textbook_chunks"
    ):
        """
        initialize the retriever.
        
        args:
            db_path: path to chromadb persistence directory
            collection_name: name of the collection to query
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"chromadb not found at: {db_path}")
        
        logger.info(f"connecting to chromadb at: {db_path}")
        
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"connected to collection '{collection_name}' with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"failed to connect to chromadb: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        retrieve top-k most relevant chunks for a query.
        
        args:
            query: the search query
            top_k: number of chunks to retrieve
            
        returns:
            list of relevant text chunks
        """
        if not query.strip():
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count())
            )
            
            documents = results.get("documents", [[]])[0]
            logger.debug(f"retrieved {len(documents)} chunks for query: {query[:50]}...")
            
            return documents
            
        except Exception as e:
            logger.warning(f"retrieval failed: {e}")
            return []
    
    def get_document_count(self) -> int:
        """return the number of documents in the collection."""
        return self.collection.count()


class ProgressTracker:
    """
    track generation progress for resume capability.
    
    stores progress in a json file alongside the output,
    allowing interrupted runs to resume from the last completed slide.
    
    attributes:
        progress_file: path to the progress tracking file
        completed_slides: set of completed slide numbers
    """
    
    def __init__(self, output_path: str):
        """
        initialize the progress tracker.
        
        args:
            output_path: path to the output tex file
        """
        self.progress_file = Path(output_path).with_suffix('.progress.json')
        self.completed_slides: set = set()
        
        # load existing progress if any
        self._load_progress()
    
    def _load_progress(self) -> None:
        """load progress from file if it exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_slides = set(data.get("completed_slides", []))
                    logger.info(f"loaded progress: {len(self.completed_slides)} slides completed")
            except Exception as e:
                logger.warning(f"failed to load progress: {e}")
    
    def _save_progress(self) -> None:
        """save progress to file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    "completed_slides": list(self.completed_slides),
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"failed to save progress: {e}")
    
    def mark_completed(self, slide_number: int) -> None:
        """mark a slide as completed."""
        self.completed_slides.add(slide_number)
        self._save_progress()
    
    def is_completed(self, slide_number: int) -> bool:
        """check if a slide has been completed."""
        return slide_number in self.completed_slides
    
    def reset(self) -> None:
        """reset all progress."""
        self.completed_slides = set()
        if self.progress_file.exists():
            self.progress_file.unlink()


class ChapterStateTracker:
    """
    tracks chapter state to prevent repetition and hallucinations.
    
    maintains:
    - list of section titles already written
    - key topics/concepts already covered
    - definitions already provided
    - examples already given
    
    this helps the llm avoid repeating content.
    """
    
    def __init__(self):
        self.sections: List[str] = []
        self.subsections: List[str] = []
        self.topics_covered: set = set()
        self.definitions_given: set = set()
        self.examples_given: set = set()
        self.figure_labels: set = set()
    
    def extract_from_content(self, content: str) -> None:
        """extract structural elements from generated content."""
        # extract section titles
        section_matches = re.findall(r'\\section\{([^}]+)\}', content)
        self.sections.extend(section_matches)
        
        # extract subsection titles
        subsection_matches = re.findall(r'\\subsection\{([^}]+)\}', content)
        self.subsections.extend(subsection_matches)
        
        # extract definition names
        def_matches = re.findall(r'\\begin\{definition\}[^{]*\{([^}]+)\}', content)
        self.definitions_given.update(def_matches)
        
        # extract example titles
        ex_matches = re.findall(r'\\begin\{example\}[^{]*\{([^}]+)\}', content)
        self.examples_given.update(ex_matches)
        
        # extract figure labels
        fig_matches = re.findall(r'\\label\{(fig:[^}]+)\}', content)
        self.figure_labels.update(fig_matches)
        
        # extract key topics from keyconcept boxes
        topic_matches = re.findall(r'\\begin\{keyconcept\}.*?\\end\{keyconcept\}', content, re.DOTALL)
        for topic in topic_matches:
            # extract first significant words
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', topic)
            self.topics_covered.update(words[:3])
    
    def get_context_summary(self) -> str:
        """generate a summary of what has been covered."""
        parts = []
        
        if self.sections:
            parts.append(f"SECTIONS ALREADY WRITTEN: {', '.join(self.sections[-5:])}")
        
        if self.subsections:
            parts.append(f"SUBSECTIONS ALREADY WRITTEN: {', '.join(self.subsections[-8:])}")
        
        if self.definitions_given:
            parts.append(f"DEFINITIONS ALREADY PROVIDED: {', '.join(list(self.definitions_given)[-10:])}")
        
        if self.examples_given:
            parts.append(f"EXAMPLES ALREADY GIVEN: {', '.join(list(self.examples_given)[-5:])}")
        
        return "\n".join(parts)
    
    def is_duplicate_section(self, title: str) -> bool:
        """check if a section title is too similar to existing ones."""
        title_lower = title.lower().strip()
        for existing in self.sections:
            existing_lower = existing.lower().strip()
            # check for exact match or high similarity
            if title_lower == existing_lower:
                return True
            # check for similar starts (e.g., "Rationality" vs "Rationality in Action")
            if title_lower.startswith(existing_lower[:20]) or existing_lower.startswith(title_lower[:20]):
                if len(title_lower) > 10 and len(existing_lower) > 10:
                    return True
        return False


class TextbookAgent:
    """
    main orchestrator for textbook generation.
    
    implements the rolling context rag pattern:
    1. for each slide, generate a query
    2. retrieve relevant textbook chunks
    3. build prompt with rolling context (last N chars of output)
    4. call llm for latex generation
    5. append to output file
    
    usage:
        agent = TextbookAgent(
            slides_json="slides.json",
            db_path="./chroma_db",
            output_path="output.tex"
        )
        agent.run()
    """
    
    # system prompt for the llm
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
        rolling_context_chars: int = 2500,  # increased for rtx 5090
        retrieval_top_k: int = 5,
        resume: bool = True
    ):
        """
        initialize the textbook agent.
        
        args:
            slides_json: path to parsed slides json
            db_path: path to chromadb persistence
            collection_name: name of the chromadb collection
            output_path: path for latex output
            model: ollama model to use
            ollama_url: ollama api base url
            rolling_context_chars: characters of previous output to include
            retrieval_top_k: number of chunks to retrieve per slide
            resume: if true, resume from last completed slide
        """
        self.slides_json = slides_json
        self.output_path = output_path
        self.rolling_context_chars = rolling_context_chars
        self.retrieval_top_k = retrieval_top_k
        self.resume = resume
        
        # initialize components
        logger.info("initializing textbook agent components...")
        
        self.slides: List[SlideData] = self._load_slides()
        self.retriever = ChromaRetriever(db_path, collection_name)
        self.ollama = OllamaClient(ollama_url, model)
        self.progress = ProgressTracker(output_path)
        self.chapter_state = ChapterStateTracker()  # track what's been written
        
        # current rolling context buffer
        self.rolling_context = ""
    
    def _load_slides(self) -> List[SlideData]:
        """load slides from json file."""
        slides_path = Path(self.slides_json)
        
        if not slides_path.exists():
            raise FileNotFoundError(f"slides json not found: {self.slides_json}")
        
        with open(slides_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        slides = [SlideData.from_dict(s) for s in data]
        logger.info(f"loaded {len(slides)} slides from {self.slides_json}")
        
        return slides
    
    def _get_rolling_context(self) -> str:
        """get the last N characters of generated output."""
        if not self.rolling_context:
            return ""
        
        return self.rolling_context[-self.rolling_context_chars:]
    
    def _update_rolling_context(self, new_content: str) -> None:
        """update the rolling context buffer with new content."""
        self.rolling_context += new_content
        
        # keep buffer from growing too large
        if len(self.rolling_context) > self.rolling_context_chars * 2:
            self.rolling_context = self.rolling_context[-self.rolling_context_chars:]
    
    def _build_prompt(
        self,
        slide: SlideData,
        retrieved_chunks: List[str]
    ) -> str:
        """
        build the generation prompt for a slide.
        
        prompt structure:
        1. chapter state (what's already covered - anti-repetition)
        2. rolling context (previous output for flow)
        3. retrieved textbook chunks (source material)
        4. current slide content (target to expand)
        
        args:
            slide: the current slide to process
            retrieved_chunks: relevant textbook chunks
            
        returns:
            formatted prompt string
        """
        parts = []
        
        # section 0: what has already been covered (anti-repetition)
        chapter_summary = self.chapter_state.get_context_summary()
        if chapter_summary:
            parts.append("=== ALREADY COVERED (DO NOT REPEAT) ===")
            parts.append(chapter_summary)
            parts.append("")
        
        # section 1: rolling context for narrative flow
        rolling = self._get_rolling_context()
        if rolling:
            parts.append("=== PREVIOUS OUTPUT (for continuity) ===")
            parts.append(rolling)
            parts.append("")
        
        # section 2: retrieved textbook content (remove "Source X" labels)
        if retrieved_chunks:
            parts.append("=== REFERENCE MATERIAL ===")
            for chunk in retrieved_chunks:
                parts.append(chunk[:1500])
                parts.append("---")
            parts.append("")
        
        # section 3: slide content to expand
        parts.append("=== SLIDE CONTENT TO EXPAND ===")
        parts.append(f"Slide {slide.slide_number}: {slide.title}")
        parts.append("")
        
        # detect slide type
        is_title_slide = self._is_title_or_outline_slide(slide)
        
        if is_title_slide:
            parts.append("NOTE: This is a TITLE/OUTLINE slide. Write only a brief introduction (2-3 sentences).")
        else:
            parts.append("Content to cover:")
            parts.append(slide.content)
        parts.append("")
        
        # instructions
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
    
    def _is_title_or_outline_slide(self, slide: SlideData) -> bool:
        """detect if a slide is just a title or outline (minimal content)."""
        content = slide.content.strip()
        
        # empty or very short content
        if len(content) < 50:
            return True
        
        # count bullet points / lines
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        if len(lines) <= 2:
            return True
        
        # check for outline keywords
        outline_keywords = ['outline', 'overview', 'agenda', 'contents', 'topics']
        title_lower = slide.title.lower()
        if any(kw in title_lower for kw in outline_keywords):
            return True
        
        return False
    
    def _initialize_output_file(self) -> None:
        """
        initialize the output tex file with preamble.
        
        if resuming and file exists, load rolling context from it.
        otherwise, create new file with preamble.
        """
        output_path = Path(self.output_path)
        
        if self.resume and output_path.exists():
            # load existing content for rolling context
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # extract content after \begin{document} for context
            doc_start = content.find(r"\begin{document}")
            if doc_start != -1:
                self.rolling_context = content[doc_start + len(r"\begin{document}"):]
                
            logger.info(f"resuming from existing file, loaded {len(self.rolling_context)} chars of context")
        else:
            # create new file with preamble
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_PREAMBLE)
            
            logger.info(f"created new output file with preamble: {output_path}")
    
    def _append_to_output(self, content: str) -> None:
        """append generated content to output file."""
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(content)
            f.write("\n\n")
    
    def _finalize_output(self) -> None:
        """add closing tags to the latex document."""
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write("\n\\end{document}\n")
        
        logger.info("added document closing tag")
    
    def _process_slide(self, slide: SlideData) -> bool:
        """
        process a single slide.
        
        args:
            slide: the slide to process
            
        returns:
            true if successful, false otherwise
        """
        console.print(f"\n[bold cyan]processing slide {slide.slide_number}:[/] {slide.title[:50]}...")
        
        try:
            # step 1: generate retrieval query
            query = slide.get_query()
            
            # step 2: retrieve relevant chunks
            chunks = self.retriever.retrieve(query, self.retrieval_top_k)
            logger.debug(f"retrieved {len(chunks)} chunks for slide {slide.slide_number}")
            
            # step 3: build prompt
            prompt = self._build_prompt(slide, chunks)
            
            # step 4: generate content
            console.print("[dim]generating content...[/dim]")
            
            generated = self.ollama.generate(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                stream=False
            )
            
            if not generated.strip():
                logger.warning(f"empty response for slide {slide.slide_number}")
                return False
            
            # step 5: clean the output
            generated = self._clean_latex_output(generated)
            
            # step 5.5: detect and remove duplicate sections
            generated = self._remove_duplicate_sections(generated)
            
            # step 6: update chapter state BEFORE appending
            self.chapter_state.extract_from_content(generated)
            
            # step 7: append to file
            self._append_to_output(generated)
            
            # step 8: update rolling context
            self._update_rolling_context(generated)
            
            # step 9: mark progress
            self.progress.mark_completed(slide.slide_number)
            
            console.print(f"[green]✓ slide {slide.slide_number} complete ({len(generated)} chars)[/green]")
            return True
            
        except Exception as e:
            logger.error(f"failed to process slide {slide.slide_number}: {e}")
            return False
    
    def _clean_latex_output(self, text: str) -> str:
        """
        clean generated latex output.
        
        removes unwanted artifacts from llm output:
        - markdown code blocks
        - document preamble if included
        - extra whitespace
        
        args:
            text: raw generated text
            
        returns:
            cleaned latex content
        """
        # remove markdown code blocks
        text = re.sub(r'```latex\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # remove any documentclass or begin{document} if llm included them
        text = re.sub(r'\\documentclass.*?\n', '', text)
        text = re.sub(r'\\begin\{document\}', '', text)
        text = re.sub(r'\\end\{document\}', '', text)
        text = re.sub(r'\\usepackage.*?\n', '', text)
        
        # remove hallucinated references
        # "Source 1", "Source 2", etc.
        text = re.sub(r'\[?Source\s*\d+\]?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\(see Source\s*\d+\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'as discussed in Source\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'according to Source\s*\d+', '', text, flags=re.IGNORECASE)
        
        # remove references to non-existent figures (unless we know they exist)
        known_figures = self.chapter_state.figure_labels if hasattr(self, 'chapter_state') else set()
        def check_figure_ref(match):
            label = match.group(1) if match.group(1) else match.group(0)
            if f"fig:{label}" not in known_figures and label not in known_figures:
                return ""  # remove invalid reference
            return match.group(0)
        
        # remove Figure~\ref{fig:X} if fig:X not defined
        text = re.sub(r'Figure~?\\ref\{fig:[^}]+\}', '', text)
        text = re.sub(r'\(see Figure~?\\ref\{[^}]+\}\)', '', text)
        
        # remove "discussed in Section X" if it looks like a hallucination
        text = re.sub(r'discussed in Section\s*\d+(\.\d+)?', 'discussed earlier', text)
        
        # normalize whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # remove orphaned parentheses from removed content
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    def _remove_duplicate_sections(self, text: str) -> str:
        """
        detect duplicate sections and log warnings (but don't remove).
        
        the ai has context about what's covered, so if it still writes
        a section, we trust its judgment but log for review.
        """
        # just detect and warn, don't delete
        section_matches = re.findall(r'\\section\{([^}]+)\}', text)
        for title in section_matches:
            if self.chapter_state.is_duplicate_section(title):
                logger.warning(f"potential duplicate section detected: {title}")
        
        subsection_matches = re.findall(r'\\subsection\{([^}]+)\}', text)
        for title in subsection_matches:
            title_lower = title.lower()
            for existing in self.chapter_state.subsections:
                if title_lower == existing.lower():
                    logger.warning(f"potential duplicate subsection: {title}")
                    break
        
        return text  # return unchanged, trust the ai
    
    def run(self) -> None:
        """
        run the full generation pipeline.
        
        processes all slides in sequence, maintaining rolling context
        and supporting resume from interruption.
        """
        console.print("\n[bold green]===== textbook generation agent =====[/]\n")
        
        # check ollama health
        console.print("[bold]step 1:[/] checking ollama connection...")
        if not self.ollama.check_health():
            console.print("[bold red]error: ollama not available[/]")
            console.print(f"[dim]ensure ollama is running with model: {self.ollama.model}[/dim]")
            console.print(f"[dim]run: ollama pull {self.ollama.model}[/dim]")
            return
        
        console.print("[green]✓ ollama connected[/green]")
        
        # initialize output file
        console.print("[bold]step 2:[/] initializing output file...")
        self._initialize_output_file()
        
        # count slides to process
        slides_to_process = [
            s for s in self.slides
            if not (self.resume and self.progress.is_completed(s.slide_number))
        ]
        
        if not slides_to_process:
            console.print("[yellow]all slides already processed![/yellow]")
            return
        
        console.print(f"[bold]step 3:[/] processing {len(slides_to_process)} slides...")
        console.print(f"[dim]chromadb: {self.retriever.get_document_count()} chunks available[/dim]")
        
        # process each slide
        successful = 0
        failed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "generating textbook...",
                total=len(slides_to_process)
            )
            
            for slide in slides_to_process:
                if self._process_slide(slide):
                    successful += 1
                else:
                    failed += 1
                
                progress.update(task, advance=1)
        
        # finalize
        console.print("\n[bold]step 4:[/] finalizing document...")
        self._finalize_output()
        
        # summary
        console.print("\n[bold green]===== generation complete =====[/]\n")
        console.print(f"[bold]slides processed:[/] {successful}/{len(slides_to_process)}")
        if failed > 0:
            console.print(f"[bold red]slides failed:[/] {failed}")
        console.print(f"[bold]output file:[/] {self.output_path}")
        
        # show file size
        output_size = Path(self.output_path).stat().st_size
        console.print(f"[bold]output size:[/] {output_size:,} bytes")


def create_argparser() -> argparse.ArgumentParser:
    """create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="generate latex textbook from slides using rolling context rag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    python main_agent.py
    python main_agent.py --slides lecture.json --output chapter1.tex
    python main_agent.py --model llama3:8b --top-k 10
    python main_agent.py --no-resume  # start fresh
        """
    )
    
    parser.add_argument(
        "--slides", "-s",
        type=str,
        default="slides.json",
        help="path to parsed slides json (default: slides.json)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.tex",
        help="output tex file path (default: output.tex)"
    )
    
    parser.add_argument(
        "--db-path", "-d",
        type=str,
        default="./chroma_db",
        help="path to chromadb persistence (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default="textbook_chunks",
        help="chromadb collection name (default: textbook_chunks)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek-r1:32b",
        help="ollama model to use (default: deepseek-r1:32b)"
    )
    
    parser.add_argument(
        "--ollama-url", "-u",
        type=str,
        default="http://localhost:11434",
        help="ollama api url (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--context-chars",
        type=int,
        default=1000,
        help="characters of previous output for rolling context (default: 1000)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="number of chunks to retrieve per slide (default: 5)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="start fresh, don't resume from previous run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="enable verbose debug logging"
    )
    
    return parser


def main():
    """main entry point for textbook generation."""
    parser = create_argparser()
    args = parser.parse_args()
    
    # set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # validate inputs exist
    if not Path(args.slides).exists():
        console.print(f"[bold red]error:[/] slides json not found: {args.slides}")
        console.print("[dim]run parse_slides.py first to create slides.json[/dim]")
        sys.exit(1)
    
    if not Path(args.db_path).exists():
        console.print(f"[bold red]error:[/] chromadb not found at: {args.db_path}")
        console.print("[dim]run setup_rag.py first to create the vector store[/dim]")
        sys.exit(1)
    
    try:
        # create and run agent
        agent = TextbookAgent(
            slides_json=args.slides,
            db_path=args.db_path,
            collection_name=args.collection,
            output_path=args.output,
            model=args.model,
            ollama_url=args.ollama_url,
            rolling_context_chars=args.context_chars,
            retrieval_top_k=args.top_k,
            resume=not args.no_resume
        )
        
        agent.run()
        
    except FileNotFoundError as e:
        console.print(f"[bold red]file error:[/] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]generation interrupted by user[/yellow]")
        console.print("[dim]progress saved, run again to resume[/dim]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]error:[/] {e}")
        logger.exception("generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
