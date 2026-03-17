# Gemini Backend

Use the Gemini backend when you want browser-automated generation instead of local inference.

## First-Time Login

```bash
python -m sliderag gemini-login
```

Or with the compatibility wrapper:

```bash
python gemini_agent.py --login
```

## Basic Usage

```bash
python -m sliderag generate --backend gemini -- \
  --slides slides.json \
  --db-path ./chroma_db \
  --output output.tex \
  --batch-size 5
```

## Important Flags

- `--batch-size`: slides processed per Gemini request
- `--browser-profile`: persistent browser profile directory
- `--response-timeout`: wait time for Gemini output
- `--request-delay`: inter-request delay for rate limiting
- `--headless`: run browser without a visible window
- `--diagnose`: inspect selectors if Google changes the UI

## Diagnostics

```bash
python -m sliderag gemini-diagnose
```

Or:

```bash
python gemini_agent.py --diagnose
```
