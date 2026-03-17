# sliderag

sliderag turns a slide deck and a reference textbook into a long-form latex chapter.

the idea is simple:

- slides give the structure
- the textbook gives the factual grounding
- the generator expands short bullet points into readable textbook prose

the output is not a summary. it is a chapter-writing pipeline built for producing detailed course material from sparse lecture slides.

## what it does

sliderag runs a rolling-context rag pipeline:

1. it reads a textbook pdf and stores chunked embeddings in chromadb
2. it reads a slide deck pdf and converts it into structured json
3. for each slide, or batch of slides, it retrieves relevant textbook context
4. it feeds the slide content, retrieved context, and recent generated output into an llm
5. it writes latex incrementally so long runs can resume after interruption

the result is a latex textbook chapter that follows the slide sequence but has much more depth, continuity, and explanation than the slides alone.

## where it fits

sliderag is useful when you have:

- lecture slides that are too brief to stand alone
- a textbook or reference pdf that contains the missing detail
- a need to produce notes, chapters, or expanded course handouts in latex

it is not a general chatbot app. it is a document-generation pipeline with a specific input shape and a specific output target.

## backends

sliderag supports two generation backends:

### ollama

use this when you want local inference or a remote machine you control.

- backend: local ollama api
- good for: private runs, local experimentation, controlled environments
- requirements: ollama running with a compatible model

example:

```bash
ollama serve &
ollama pull deepseek-r1:32b
python -m sliderag generate --backend ollama -- --slides runs/course/slides.json
```

### gemini

use this when you want browser-driven generation through the gemini web app.

- backend: gemini in a persistent chromium profile
- good for: avoiding local inference setup, using gemini through a saved session
- requirements: playwright, chromium, and a working google login

example:

```bash
python -m sliderag gemini-login
python -m sliderag generate --backend gemini -- --slides runs/course/slides.json
```

more detail:

- [docs/backends/ollama.md](docs/backends/ollama.md)
- [docs/backends/gemini.md](docs/backends/gemini.md)

## install

```bash
pip install -r requirements.txt
```

for the gemini backend:

```bash
playwright install chromium
```

## core workflow

### 1. build the rag index

ingest the reference textbook into chromadb:

```bash
python -m sliderag setup-rag -- \
  --pdf data/input/textbook.pdf \
  --db-path data/indexes/chroma_db
```

### 2. parse the slides

extract slide titles and content into structured json:

```bash
python -m sliderag parse-slides -- \
  --pdf data/input/slides.pdf \
  --output runs/course/slides.json
```

### 3. generate the chapter

with ollama:

```bash
python -m sliderag generate --backend ollama -- \
  --slides runs/course/slides.json \
  --db-path data/indexes/chroma_db \
  --output runs/course/output.tex \
  --model deepseek-r1:32b
```

with gemini:

```bash
python -m sliderag generate --backend gemini -- \
  --slides runs/course/slides.json \
  --db-path data/indexes/chroma_db \
  --output runs/course/output.tex \
  --batch-size 5 \
  --browser-profile data/profiles/gemini_browser_profile
```

## cli

the main entrypoint is:

```bash
python -m sliderag
```

available commands:

```bash
python -m sliderag help generate
python -m sliderag parse-slides -- --pdf slides.pdf
python -m sliderag setup-rag -- --pdf textbook.pdf
python -m sliderag generate --backend ollama -- --slides slides.json
python -m sliderag generate --backend gemini -- --slides slides.json
python -m sliderag gemini-login
python -m sliderag gemini-diagnose
```

the `--` separator passes the remaining flags directly to the underlying module.

## inputs and outputs

recommended layout:

```text
data/
├── input/
│   ├── slides.pdf
│   └── textbook.pdf
├── indexes/
│   └── chroma_db/
└── profiles/
    └── gemini_browser_profile/

runs/
└── course-name/
    ├── slides.json
    ├── output.tex
    └── output.progress.json
```

main generated artifact:

- `output.tex`: the generated chapter in latex

other runtime artifacts:

- `slides.json`: parsed slides
- `*.progress.json`: resume state
- `chroma_db/`: vector index
- `gemini_browser_profile/`: persistent browser session for gemini

## compatibility commands

the old entrypoints still work:

```bash
python main_agent.py
python gemini_agent.py --login
python parse_slides.py --pdf slides.pdf
python setup_rag.py --pdf textbook.pdf
```

## environment note

the embedding stack currently expects `numpy<2`, and `requirements.txt` pins that. if your environment already has numpy 2 with older compiled ml wheels, rebuild the venv before running `setup-rag`.
