#!/usr/bin/env python3
"""
setup_rag.py - textbook ingestion and chromadb vector store setup

this module handles:
1. loading and extracting text from textbook.pdf
2. intelligent chunking with overlap for context preservation
3. generating embeddings using sentence-transformers
4. storing embeddings in persistent chromadb instance

usage:
    python setup_rag.py [--pdf textbook.pdf] [--db-path ./chroma_db] [--chunk-size 512] [--overlap 50]

author: sliderag project
license: mit
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sentence_transformers import SentenceTransformer
import fitz  # pymupdf

# configure rich console for beautiful output
console = Console()

# configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("setup_rag")


class TextChunker:
    """
    intelligent text chunking with configurable size and overlap.
    
    this class implements a sliding window approach to chunking that:
    - respects sentence boundaries where possible
    - maintains configurable overlap between chunks for context continuity
    - handles edge cases like very short documents or single paragraphs
    
    attributes:
        chunk_size: target number of characters per chunk
        chunk_overlap: number of characters to overlap between adjacent chunks
        sentence_endings: tuple of characters that indicate sentence boundaries
    """
    
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        respect_sentences: bool = True
    ):
        """
        initialize the text chunker.
        
        args:
            chunk_size: target characters per chunk (default 1500 for ~350-400 tokens)
            chunk_overlap: overlap characters between chunks (default 200)
            respect_sentences: if true, try to break at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentences = respect_sentences
        self.sentence_endings = ('.', '!', '?', '。', '！', '？')
        
        logger.info(f"initialized chunker: size={chunk_size}, overlap={chunk_overlap}")
    
    def _find_sentence_boundary(self, text: str, target_pos: int, search_range: int = 100) -> int:
        """
        find the nearest sentence boundary to the target position.
        
        searches within search_range characters before target_pos for a sentence ending.
        if no sentence boundary found, returns target_pos.
        
        args:
            text: the text to search in
            target_pos: the ideal position to break at
            search_range: how far back to search for sentence boundary
            
        returns:
            the position of the sentence boundary, or target_pos if none found
        """
        if not self.respect_sentences:
            return target_pos
            
        # clamp search range to available text
        search_start = max(0, target_pos - search_range)
        search_text = text[search_start:target_pos]
        
        # find last sentence ending in search range
        best_pos = -1
        for ending in self.sentence_endings:
            pos = search_text.rfind(ending)
            if pos > best_pos:
                best_pos = pos
        
        if best_pos != -1:
            # return absolute position (after the sentence ending)
            return search_start + best_pos + 1
        
        # no sentence boundary found, try to break at whitespace
        last_space = search_text.rfind(' ')
        if last_space != -1:
            return search_start + last_space + 1
            
        return target_pos
    
    def chunk(self, text: str) -> List[str]:
        """
        split text into overlapping chunks.
        
        this method implements a sliding window chunking algorithm:
        1. start at position 0
        2. find the end position (start + chunk_size)
        3. adjust end to nearest sentence boundary
        4. extract chunk from start to end
        5. move start forward by (chunk_size - overlap)
        6. repeat until end of text
        
        args:
            text: the full text to chunk
            
        returns:
            list of text chunks with overlap
        """
        if not text or not text.strip():
            logger.warning("empty text provided to chunker")
            return []
        
        # normalize whitespace
        text = ' '.join(text.split())
        
        # if text is shorter than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            logger.info("text shorter than chunk size, returning as single chunk")
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # calculate ideal end position
            end = min(start + self.chunk_size, text_len)
            
            # if not at end of text, find sentence boundary
            if end < text_len:
                end = self._find_sentence_boundary(text, end)
            
            # extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # move start forward, accounting for overlap
            # but ensure we make progress (at least 100 chars)
            step = max(100, self.chunk_size - self.chunk_overlap)
            start = start + step
            
            # avoid infinite loop if step is too small
            if start >= text_len - 50:
                break
        
        logger.info(f"created {len(chunks)} chunks from {text_len} characters")
        return chunks


class PDFExtractor:
    """
    robust pdf text extraction using pymupdf (fitz).
    
    this class provides:
    - full document text extraction
    - page-by-page extraction with metadata
    - handling of complex layouts
    - unicode normalization
    
    attributes:
        pdf_path: path to the pdf file
        doc: the opened pymupdf document
    """
    
    def __init__(self, pdf_path: str):
        """
        initialize the pdf extractor.
        
        args:
            pdf_path: path to the pdf file to extract
            
        raises:
            filenotfounderror: if pdf file does not exist
            exception: if pdf cannot be opened
        """
        self.pdf_path = Path(pdf_path)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"pdf not found: {pdf_path}")
        
        if not self.pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"file is not a pdf: {pdf_path}")
        
        try:
            self.doc = fitz.open(str(self.pdf_path))
            logger.info(f"opened pdf: {pdf_path} ({len(self.doc)} pages)")
        except Exception as e:
            logger.error(f"failed to open pdf: {e}")
            raise
    
    def extract_full_text(self) -> str:
        """
        extract all text from the pdf as a single string.
        
        iterates through all pages and concatenates their text content.
        applies basic text cleaning to handle common pdf extraction issues.
        
        returns:
            the complete text content of the pdf
        """
        full_text = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("extracting text from pdf...", total=len(self.doc))
            
            for page_num, page in enumerate(self.doc):
                try:
                    # extract text with better layout preservation
                    text = page.get_text("text")
                    
                    # basic cleaning
                    text = self._clean_text(text)
                    
                    if text.strip():
                        full_text.append(text)
                        
                except Exception as e:
                    logger.warning(f"failed to extract page {page_num + 1}: {e}")
                
                progress.update(task, advance=1)
        
        combined = "\n\n".join(full_text)
        logger.info(f"extracted {len(combined)} characters from {len(self.doc)} pages")
        return combined
    
    def extract_by_page(self) -> List[Tuple[int, str]]:
        """
        extract text from each page with page numbers.
        
        useful for maintaining page references in chunks for citation purposes.
        
        returns:
            list of tuples (page_number, page_text)
        """
        pages = []
        
        for page_num, page in enumerate(self.doc):
            try:
                text = page.get_text("text")
                text = self._clean_text(text)
                if text.strip():
                    pages.append((page_num + 1, text))
            except Exception as e:
                logger.warning(f"failed to extract page {page_num + 1}: {e}")
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """
        clean extracted text to fix common pdf extraction issues.
        
        handles:
        - excessive whitespace
        - hyphenated line breaks
        - unicode normalization
        - control characters
        
        args:
            text: raw extracted text
            
        returns:
            cleaned text
        """
        if not text:
            return ""
        
        # replace common problematic characters
        replacements = {
            '\x00': '',      # null bytes
            '\x0c': '\n',    # form feed
            '\x0b': '\n',    # vertical tab
            '\xa0': ' ',     # non-breaking space
            '­': '',         # soft hyphen
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # fix hyphenated line breaks (word- \n continuation)
        import re
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # normalize multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # normalize multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def get_document_hash(self) -> str:
        """
        compute md5 hash of the pdf file.
        
        useful for checking if the document has changed and needs re-indexing.
        
        returns:
            md5 hash string of the file contents
        """
        hasher = hashlib.md5()
        with open(self.pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def close(self):
        """close the pdf document."""
        if self.doc:
            self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class EmbeddingGenerator:
    """
    generate embeddings using sentence-transformers.
    
    uses the all-MiniLM-L6-v2 model by default which provides a good balance
    of speed and quality for semantic search applications.
    
    attributes:
        model_name: the sentence-transformer model to use
        model: the loaded sentence-transformer model
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        initialize the embedding generator.
        
        args:
            model_name: name of the sentence-transformer model to use
        """
        self.model_name = model_name
        
        logger.info(f"loading embedding model: {model_name}")
        console.print(f"[bold blue]loading embedding model:[/] {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"model loaded, embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"failed to load embedding model: {e}")
            raise
    
    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        generate embeddings for a list of texts.
        
        args:
            texts: list of text strings to embed
            batch_size: number of texts to embed at once
            show_progress: whether to show a progress bar
            
        returns:
            list of embedding vectors (as lists of floats)
        """
        if not texts:
            return []
        
        logger.info(f"generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # convert numpy arrays to lists for chromadb compatibility
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"embedding generation failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """return the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()


class ChromaDBManager:
    """
    manage chromadb vector store operations.
    
    handles:
    - persistent chromadb client initialization
    - collection creation and management
    - document storage with metadata
    - idempotent operations (skip if already indexed)
    
    attributes:
        db_path: path to the chromadb persistence directory
        client: the chromadb persistent client
        collection_name: name of the vector collection
    """
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "textbook_chunks"
    ):
        """
        initialize the chromadb manager.
        
        args:
            db_path: path for chromadb persistence
            collection_name: name of the collection to use
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        
        # create db directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"initializing chromadb at: {self.db_path}")
        
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("chromadb client initialized successfully")
        except Exception as e:
            logger.error(f"failed to initialize chromadb: {e}")
            raise
    
    def collection_exists(self) -> bool:
        """check if the collection already exists."""
        try:
            collections = self.client.list_collections()
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False
    
    def get_collection(self, embedding_dimension: Optional[int] = None) -> chromadb.Collection:
        """
        get or create the vector collection.
        
        args:
            embedding_dimension: dimension of embeddings (for metadata)
            
        returns:
            the chromadb collection object
        """
        try:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "textbook chunks for rag retrieval",
                    "embedding_dimension": str(embedding_dimension) if embedding_dimension else "unknown"
                }
            )
            logger.info(f"collection '{self.collection_name}' ready, {collection.count()} documents")
            return collection
        except Exception as e:
            logger.error(f"failed to get/create collection: {e}")
            raise
    
    def add_documents(
        self,
        collection: chromadb.Collection,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        add documents to the collection.
        
        args:
            collection: the chromadb collection
            chunks: list of text chunks
            embeddings: list of embedding vectors
            metadatas: optional list of metadata dicts
            ids: optional list of document ids
        """
        if not chunks:
            logger.warning("no chunks to add")
            return
        
        # generate ids if not provided
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # generate default metadata if not provided
        if metadatas is None:
            metadatas = [{"chunk_index": i, "chunk_length": len(c)} for i, c in enumerate(chunks)]
        
        logger.info(f"adding {len(chunks)} documents to collection")
        
        # chromadb has a batch size limit, so we add in batches
        batch_size = 100
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("storing in chromadb...", total=len(chunks))
            
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                
                try:
                    collection.add(
                        ids=ids[i:batch_end],
                        documents=chunks[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        metadatas=metadatas[i:batch_end]
                    )
                except Exception as e:
                    logger.error(f"failed to add batch {i}-{batch_end}: {e}")
                    raise
                
                progress.update(task, advance=batch_end - i)
        
        logger.info(f"added {len(chunks)} documents, collection now has {collection.count()} total")
    
    def check_document_hash(self, collection: chromadb.Collection, doc_hash: str) -> bool:
        """
        check if a document with the given hash is already indexed.
        
        args:
            collection: the chromadb collection
            doc_hash: md5 hash of the source document
            
        returns:
            true if document is already indexed
        """
        try:
            # check collection metadata for stored hash
            metadata = collection.metadata
            stored_hash = metadata.get("source_document_hash", "")
            return stored_hash == doc_hash
        except Exception:
            return False
    
    def update_collection_metadata(
        self,
        collection: chromadb.Collection,
        doc_hash: str,
        source_file: str
    ) -> None:
        """
        update collection metadata with source document info.
        
        args:
            collection: the chromadb collection
            doc_hash: hash of the source document
            source_file: path to the source document
        """
        try:
            collection.modify(
                metadata={
                    "source_document_hash": doc_hash,
                    "source_file": source_file,
                    "description": "textbook chunks for rag retrieval",
                    "embedding_dimension": str(collection.metadata.get("embedding_dimension", "unknown"))
                }
            )
            logger.info("collection metadata updated with source document info")
        except Exception as e:
            logger.warning(f"failed to update collection metadata: {e}")


class TextbookIngester:
    """
    main orchestrator for textbook ingestion pipeline.
    
    coordinates:
    1. pdf text extraction
    2. text chunking
    3. embedding generation
    4. chromadb storage
    
    provides idempotent operation - skips re-indexing if document unchanged.
    
    usage:
        ingester = TextbookIngester(
            pdf_path="textbook.pdf",
            db_path="./chroma_db"
        )
        collection = ingester.run()
    """
    
    def __init__(
        self,
        pdf_path: str,
        db_path: str = "./chroma_db",
        collection_name: str = "textbook_chunks",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        force_reindex: bool = False
    ):
        """
        initialize the textbook ingester.
        
        args:
            pdf_path: path to the textbook pdf
            db_path: path for chromadb persistence
            collection_name: name of the vector collection
            chunk_size: target characters per chunk
            chunk_overlap: overlap between chunks
            embedding_model: sentence-transformer model name
            force_reindex: if true, re-index even if document unchanged
        """
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.force_reindex = force_reindex
        
        # will be initialized during run
        self.pdf_extractor: Optional[PDFExtractor] = None
        self.chunker: Optional[TextChunker] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.db_manager: Optional[ChromaDBManager] = None
    
    def run(self) -> chromadb.Collection:
        """
        run the full ingestion pipeline.
        
        steps:
        1. extract text from pdf
        2. compute document hash for change detection
        3. check if already indexed (skip if unchanged)
        4. chunk the text
        5. generate embeddings
        6. store in chromadb
        
        returns:
            the chromadb collection containing the indexed chunks
        """
        console.print("\n[bold green]===== textbook ingestion pipeline =====[/]\n")
        
        # step 1: initialize components
        console.print("[bold]step 1:[/] initializing components...")
        
        self.pdf_extractor = PDFExtractor(self.pdf_path)
        self.chunker = TextChunker(self.chunk_size, self.chunk_overlap)
        self.db_manager = ChromaDBManager(self.db_path, self.collection_name)
        
        # step 2: compute document hash
        console.print("[bold]step 2:[/] computing document hash...")
        doc_hash = self.pdf_extractor.get_document_hash()
        logger.info(f"document hash: {doc_hash}")
        
        # step 3: check if already indexed
        collection = self.db_manager.get_collection()
        
        if not self.force_reindex and self.db_manager.check_document_hash(collection, doc_hash):
            console.print("[bold yellow]document already indexed with same hash - skipping ingestion[/]")
            console.print(f"[dim]collection has {collection.count()} chunks[/]")
            self.pdf_extractor.close()
            return collection
        
        # step 4: extract text
        console.print("[bold]step 3:[/] extracting text from pdf...")
        full_text = self.pdf_extractor.extract_full_text()
        console.print(f"[dim]extracted {len(full_text):,} characters[/]")
        
        # step 5: chunk the text
        console.print("[bold]step 4:[/] chunking text...")
        chunks = self.chunker.chunk(full_text)
        console.print(f"[dim]created {len(chunks)} chunks[/]")
        
        if not chunks:
            console.print("[bold red]no chunks created - check pdf content[/]")
            self.pdf_extractor.close()
            return collection
        
        # step 6: generate embeddings
        console.print("[bold]step 5:[/] generating embeddings...")
        self.embedding_generator = EmbeddingGenerator(self.embedding_model)
        embeddings = self.embedding_generator.embed(chunks)
        console.print(f"[dim]generated {len(embeddings)} embeddings of dimension {self.embedding_generator.get_embedding_dimension()}[/]")
        
        # step 7: clear existing collection if re-indexing
        if collection.count() > 0:
            console.print("[bold yellow]clearing existing collection for re-indexing...[/]")
            self.db_manager.client.delete_collection(self.collection_name)
            collection = self.db_manager.get_collection(self.embedding_generator.get_embedding_dimension())
        
        # step 8: store in chromadb
        console.print("[bold]step 6:[/] storing in chromadb...")
        
        # create metadata for each chunk
        metadatas = [
            {
                "chunk_index": i,
                "chunk_length": len(chunk),
                "source_file": str(self.pdf_path),
                "source_hash": doc_hash
            }
            for i, chunk in enumerate(chunks)
        ]
        
        self.db_manager.add_documents(
            collection=collection,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # step 9: update collection metadata
        self.db_manager.update_collection_metadata(collection, doc_hash, str(self.pdf_path))
        
        # cleanup
        self.pdf_extractor.close()
        
        console.print("\n[bold green]✓ ingestion complete![/]")
        console.print(f"[dim]collection '{self.collection_name}' now has {collection.count()} chunks[/]")
        console.print(f"[dim]chromadb persisted at: {self.db_path}[/]\n")
        
        return collection


def create_argparser() -> argparse.ArgumentParser:
    """create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="ingest textbook pdf into chromadb vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    python setup_rag.py
    python setup_rag.py --pdf textbook.pdf --db-path ./my_db
    python setup_rag.py --chunk-size 1000 --overlap 100 --force
        """
    )
    
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        default="textbook.pdf",
        help="path to the textbook pdf file (default: textbook.pdf)"
    )
    
    parser.add_argument(
        "--db-path", "-d",
        type=str,
        default="./chroma_db",
        help="path for chromadb persistence (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default="textbook_chunks",
        help="name of the chromadb collection (default: textbook_chunks)"
    )
    
    parser.add_argument(
        "--chunk-size", "-s",
        type=int,
        default=1500,
        help="target characters per chunk (default: 1500)"
    )
    
    parser.add_argument(
        "--overlap", "-o",
        type=int,
        default=200,
        help="character overlap between chunks (default: 200)"
    )
    
    parser.add_argument(
        "--embedding-model", "-m",
        type=str,
        default="all-MiniLM-L6-v2",
        help="sentence-transformer model name (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="force re-indexing even if document unchanged"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="enable verbose debug logging"
    )
    
    return parser


def main():
    """main entry point for textbook ingestion."""
    parser = create_argparser()
    args = parser.parse_args()
    
    # set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # validate pdf exists
    if not Path(args.pdf).exists():
        console.print(f"[bold red]error:[/] pdf file not found: {args.pdf}")
        console.print("[dim]make sure textbook.pdf exists in the current directory[/]")
        sys.exit(1)
    
    try:
        # run the ingestion pipeline
        ingester = TextbookIngester(
            pdf_path=args.pdf,
            db_path=args.db_path,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            embedding_model=args.embedding_model,
            force_reindex=args.force
        )
        
        collection = ingester.run()
        
        # print summary
        console.print("[bold]summary:[/]")
        console.print(f"  • collection: {args.collection}")
        console.print(f"  • documents: {collection.count()}")
        console.print(f"  • db path: {args.db_path}")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]file error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]error:[/] {e}")
        logger.exception("ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
