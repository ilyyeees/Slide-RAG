# Ollama Backend

Use the local LLM backend when you have Ollama running on your machine or remote box.

## Basic Usage

```bash
python -m sliderag generate --backend ollama -- \
  --slides slides.json \
  --db-path ./chroma_db \
  --output output.tex \
  --model deepseek-r1:32b
```

Compatibility wrapper:

```bash
python main_agent.py --slides slides.json --db-path ./chroma_db
```

## Important Flags

- `--model`: Ollama model name
- `--ollama-url`: Ollama API base URL
- `--context-chars`: rolling context size
- `--top-k`: retrieved chunks per slide
- `--no-resume`: start from scratch

## Example Setup

```bash
ollama serve &
ollama pull deepseek-r1:32b
python -m sliderag generate --backend ollama -- --model deepseek-r1:32b
```
