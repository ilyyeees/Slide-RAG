# sliderag

sliderag is now one project, not two drifting copies.

the shared pipeline lives in one package, and the llm choice is a backend decision:

- `ollama` for local generation through the ollama api
- `gemini` for browser-driven generation through the gemini web app

the slide parser, rag indexer, progress tracking, latex cleanup, and shared generation state all live in one place now.

## what changed

the repo used to be split between a root version and a separate `gemini version/` fork. that made every change expensive because shared logic had to be copied, compared, and fixed twice.

that is gone now.

the codebase is organized as a single package:

```text
sliderag/
├── __main__.py
├── browser_client.py
├── cli.py
├── common.py
├── gemini.py
├── ollama.py
├── parse_slides.py
└── setup_rag.py
```

the old script names still work, but they are wrappers now:

```text
main_agent.py
gemini_agent.py
parse_slides.py
setup_rag.py
```

## the new cli

the main way in is now:

```bash
python -m sliderag
```

that opens the top-level command hub with the shared workflow and every module entrypoint.

### top-level commands

```bash
python -m sliderag help generate
python -m sliderag parse-slides -- --pdf data/input/slides.pdf --output runs/course/slides.json
python -m sliderag setup-rag -- --pdf data/input/textbook.pdf --db-path data/indexes/chroma_db
python -m sliderag generate --backend ollama -- --slides runs/course/slides.json
python -m sliderag generate --backend gemini -- --slides runs/course/slides.json
python -m sliderag gemini-login
python -m sliderag gemini-diagnose
```

the `--` separator is the cleanest way to pass the rest of the flags through to the underlying module.

## install

```bash
pip install -r requirements.txt
```

for the gemini backend, install chromium once:

```bash
playwright install chromium
```

note on the python stack: the current ml dependencies need `numpy<2`, and the repo now pins that in `requirements.txt`.

## normal workflow

### 1. build the textbook index

```bash
python -m sliderag setup-rag -- \
  --pdf data/input/textbook.pdf \
  --db-path data/indexes/chroma_db
```

### 2. parse the slide deck

```bash
python -m sliderag parse-slides -- \
  --pdf data/input/slides.pdf \
  --output runs/course/slides.json
```

### 3. generate with the backend you want

ollama:

```bash
python -m sliderag generate --backend ollama -- \
  --slides runs/course/slides.json \
  --db-path data/indexes/chroma_db \
  --output runs/course/output.tex \
  --model deepseek-r1:32b
```

gemini:

```bash
python -m sliderag gemini-login

python -m sliderag generate --backend gemini -- \
  --slides runs/course/slides.json \
  --db-path data/indexes/chroma_db \
  --output runs/course/output.tex \
  --batch-size 5 \
  --browser-profile data/profiles/gemini_browser_profile
```

## backend notes

### ollama

use this when you want local inference or a remote box you control.

```bash
ollama serve &
ollama pull deepseek-r1:32b
python -m sliderag generate --backend ollama -- --model deepseek-r1:32b
```

full notes: [docs/backends/ollama.md](docs/backends/ollama.md)

### gemini

use this when you want browser automation instead of local inference.

```bash
python -m sliderag gemini-login
python -m sliderag gemini-diagnose
python -m sliderag generate --backend gemini -- --batch-size 3
```

full notes: [docs/backends/gemini.md](docs/backends/gemini.md)

## recommended repo hygiene

keep runtime state outside the source package. a sane layout looks like this:

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

the repo now ignores:

- browser profiles
- chromadb state
- generated tex and pdf files
- progress files
- the old `gemini version/` directory

## why this layout is better

- shared logic has one home, so fixes stop landing in one backend and getting forgotten in the other
- the cli makes the project feel like one tool instead of four unrelated scripts
- runtime state is easier to keep out of git
- backend choice is now explicit instead of being encoded in the folder layout

## compatibility commands

if you still want the old command names, these still forward into the new package:

```bash
python main_agent.py
python gemini_agent.py --login
python parse_slides.py --pdf slides.pdf
python setup_rag.py --pdf textbook.pdf
```

## short version

if you only remember four commands, remember these:

```bash
python -m sliderag
python -m sliderag setup-rag -- --pdf textbook.pdf
python -m sliderag parse-slides -- --pdf slides.pdf
python -m sliderag generate --backend ollama -- --slides slides.json
```
