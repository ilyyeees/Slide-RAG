# SlideRAG - Rolling Context RAG for Textbook Generation

An autonomous textbook generation agent that synthesizes course slides and textbook content into cohesive, deep-dive LaTeX chapters using a Rolling Context RAG architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ROLLING CONTEXT RAG PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────────────────┐  │
│  │ textbook.pdf │───►│ setup_rag.py  │───►│ ChromaDB (Persistent Store) │  │
│  └──────────────┘    └───────────────┘    └─────────────────────────────┘  │
│                                                         │                   │
│  ┌──────────────┐    ┌───────────────┐                  │                   │
│  │  slides.pdf  │───►│parse_slides.py│───►slides.json   │                   │
│  └──────────────┘    └───────────────┘        │         │                   │
│                                               ▼         ▼                   │
│                                    ┌─────────────────────────┐              │
│                                    │    main_agent.py        │              │
│                                    │  ┌───────────────────┐  │              │
│                                    │  │ For each slide:   │  │              │
│                                    │  │ 1. Query ChromaDB │  │              │
│                                    │  │ 2. Build prompt:  │  │              │
│                                    │  │    - Rolling ctx  │  │              │
│                                    │  │    - Retrieved    │  │              │
│                                    │  │    - Slide target │  │              │
│                                    │  │ 3. Call Ollama    │  │              │
│                                    │  │ 4. Append output  │  │              │
│                                    │  └───────────────────┘  │              │
│                                    └─────────────────────────┘              │
│                                               │                             │
│                                               ▼                             │
│                                    ┌─────────────────────────┐              │
│                                    │      output.tex         │              │
│                                    └─────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Ollama (on Vast.ai or local)

```bash
# install ollama if not already installed
curl -fsSL https://ollama.com/install.sh | sh

# start ollama server (runs in background)
ollama serve &

# pull the reasoning model (32b recommended for best quality)
ollama pull deepseek-r1:32b

# alternative models (if deepseek unavailable)
# ollama pull llama3:8b
# ollama pull mistral:7b
```

### 3. Prepare Input Files

Place your files in the project root:
- `textbook.pdf` - the reference textbook
- `slides.pdf` - course lecture slides

### 4. Run the Pipeline

```bash
# step 1: ingest textbook into vector database
python setup_rag.py --pdf textbook.pdf

# step 2: parse slides into structured json
python parse_slides.py --pdf slides.pdf

# step 3: generate the textbook
python main_agent.py
```

The generated LaTeX will be saved to `output.tex`.

## Detailed Usage

### setup_rag.py - Textbook Ingestion

Ingests the textbook PDF into a ChromaDB vector store for semantic retrieval.

```bash
python setup_rag.py [options]

options:
  --pdf, -p         path to textbook pdf (default: textbook.pdf)
  --db-path, -d     chromadb persistence path (default: ./chroma_db)
  --chunk-size, -s  target characters per chunk (default: 1500)
  --overlap, -o     overlap between chunks (default: 200)
  --force, -f       force re-indexing even if unchanged
  --verbose, -v     enable debug logging
```

**features:**
- intelligent chunking with sentence boundary detection
- embeddings using sentence-transformers (all-MiniLM-L6-v2)
- idempotent operation (skips if document unchanged)
- persistent chromadb storage

### parse_slides.py - Slide Parsing

Converts slide PDF into structured JSON for the generation loop.

```bash
python parse_slides.py [options]

options:
  --pdf, -p           path to slides pdf (default: slides.pdf)
  --output, -o        output json path (default: slides.json)
  --fallback, -f      use pdfplumber instead of pymupdf
  --include-empty     include empty slides
  --verbose, -v       enable debug logging
```

**output format:**
```json
[
  {
    "slide_number": 1,
    "title": "Introduction to AI",
    "content": "- what is ai?\n- history of ai\n- applications"
  }
]
```

### main_agent.py - Generation Agent

The core generation engine implementing Rolling Context RAG.

```bash
python main_agent.py [options]

options:
  --slides, -s        parsed slides json (default: slides.json)
  --output, -o        output tex file (default: output.tex)
  --db-path, -d       chromadb path (default: ./chroma_db)
  --model, -m         ollama model (default: deepseek-r1:32b)
  --ollama-url, -u    ollama api url (default: http://localhost:11434)
  --context-chars     rolling context size (default: 1000)
  --top-k, -k         chunks to retrieve per slide (default: 5)
  --no-resume         start fresh, don't resume
  --verbose, -v       enable debug logging
```

**features:**
- rolling context for narrative continuity
- semantic retrieval from textbook
- progress tracking with resume capability
- full latex preamble included

## Model Recommendations

| Model | Best For | Notes |
|-------|----------|-------|
| `deepseek-r1:8b` | mathematical/latex content | recommended for rigorous textbooks |
| `llama3:8b` | general content | good balance of speed/quality |
| `mistral:7b` | faster generation | lighter weight alternative |
| `deepseek-r1:32b` | highest quality | requires more vram |

## Vast.ai Deployment

### Instance Setup

1. create a vast.ai instance with gpu (rtx 3090 or better recommended)
2. select pytorch or cuda docker image
3. ssh into the instance

### Commands

```bash
# clone your repo
git clone https://github.com/yourusername/sliderag.git
cd sliderag

# install python dependencies
pip install -r requirements.txt

# install ollama
curl -fsSL https://ollama.com/install.sh | sh

# start ollama (in background with nohup)
nohup ollama serve > ollama.log 2>&1 &

# pull the model (this may take a while)
ollama pull deepseek-r1:32b

# upload your pdfs (use scp or direct upload)
# scp textbook.pdf user@vastai-ip:~/sliderag/
# scp slides.pdf user@vastai-ip:~/sliderag/

# run the pipeline
python setup_rag.py
python parse_slides.py
python main_agent.py

# download the output
# scp user@vastai-ip:~/sliderag/output.tex ./
```

### Monitoring

```bash
# watch generation progress
tail -f output.tex

# check ollama logs
tail -f ollama.log

# monitor gpu usage
nvidia-smi -l 1
```

## Troubleshooting

### Ollama Connection Failed

```bash
# check if ollama is running
curl http://localhost:11434/api/tags

# restart ollama
pkill ollama
ollama serve &
```

### Out of Memory

- use a smaller model: `--model llama3:8b` or `--model mistral:7b`
- reduce chunk retrieval: `--top-k 3`
- reduce context: `--context-chars 500`

### Empty Output

- check slides.json has content: `cat slides.json | head`
- verify chromadb has documents: run python and check `collection.count()`
- check ollama model is working: `ollama run deepseek-r1:8b "test"`

### Resume After Interruption

The agent automatically saves progress. just run again:

```bash
python main_agent.py
```

To start fresh:

```bash
python main_agent.py --no-resume
rm output.tex output.progress.json
```

## Project Structure

```
sliderag/
├── requirements.txt     # python dependencies
├── setup_rag.py         # textbook ingestion to chromadb
├── parse_slides.py      # slide pdf to json parser
├── main_agent.py        # generation agent
├── README.md            # this file
├── AGENTS.md            # operational directives
├── Instructions.md      # project specifications
│
├── textbook.pdf         # input: reference textbook (user provides)
├── slides.pdf           # input: course slides (user provides)
├── slides.json          # intermediate: parsed slides
├── output.tex           # output: generated latex
├── output.progress.json # progress tracking
│
└── chroma_db/           # persistent vector store
    ├── chroma.sqlite3
    └── ...
```

## License

MIT
