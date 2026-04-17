# LLM-Wiki — Usage Guide

A hands-on guide for using LLM-Wiki day-to-day. Start here if you want to **do things** with it rather than understand the architecture (for that, see [README.md](./README.md)).

---

## Table of Contents

- [Daily Startup](#daily-startup)
- [Shutting Down](#shutting-down)
- [The Web UI vs The CLI — When To Use Each](#the-web-ui-vs-the-cli--when-to-use-each)
- [Common Workflows](#common-workflows)
  - [Adding and ingesting a document](#adding-and-ingesting-a-document)
  - [Asking a question](#asking-a-question)
  - [Cleaning up: deleting a source](#cleaning-up-deleting-a-source)
  - [Re-ingesting a source](#re-ingesting-a-source)
  - [Browsing the wiki visually](#browsing-the-wiki-visually)
  - [Keeping the wiki healthy](#keeping-the-wiki-healthy)
- [Query Scope — Wiki vs Raw vs Hybrid](#query-scope--wiki-vs-raw-vs-hybrid)
- [Full CLI Command Reference](#full-cli-command-reference)
- [Troubleshooting](#troubleshooting)
- [First-Time Setup (If You're Starting Fresh)](#first-time-setup-if-youre-starting-fresh)

---

## Daily Startup

Every time you want to use LLM-Wiki, you need two things running: **Ollama** (the local LLM server) and **the LLM-Wiki web server** (the UI).

### 1. Make sure Ollama is running

Ollama should auto-start when your Mac boots. To check:

```bash
ollama list
```

If you see `qwen3:14b` in the list, Ollama is running and the model is available. If the command errors out, start Ollama by opening the **Ollama app** from your Applications folder, or run:

```bash
ollama serve &
```

### 2. Start the LLM-Wiki server

Open a new Terminal window and run:

```bash
cd ~/Documents/LLM-Wiki
source .venv/bin/activate
wiki serve
```

Your terminal prompt should change to start with `(LLM-Wiki)` after activation. Then `wiki serve` prints something like:

```
╭──────────────── 🚀 Serve ────────────────╮
│ LLM-Wiki web UI starting…                │
│                                          │
│   URL: http://127.0.0.1:8000             │
│   Project: /Users/you/Documents/LLM-Wiki │
│                                          │
│ Press Ctrl+C to stop.                    │
╰──────────────────────────────────────────╯
```

Your browser should open to `http://127.0.0.1:8000` automatically. If not, open it manually.

### 3. (Optional) Open Obsidian

If you use Obsidian for browsing the wiki visually, open the `wiki/` subfolder as a vault. You'll see the same content the web UI shows, plus Obsidian's graph view, backlinks, and live preview.

---

## Shutting Down

- **To stop the server:** Press `Ctrl+C` in the terminal where `wiki serve` is running.
- **To exit the venv:** Type `deactivate` (rarely needed — you can just close the terminal).
- **To stop Ollama:** Usually not necessary. If you want to free up RAM, quit the Ollama app from the menu bar icon.

---

## The Web UI vs The CLI — When To Use Each

Most things can be done in both. Quick guide:

| Task | Best tool | Why |
|---|---|---|
| Browsing sources, viewing wiki pages | **Web UI** | Prettier, with links and preview |
| Asking a question | **Web UI** | See hits panel, save as synthesis |
| Ingesting 1-2 files occasionally | **Web UI** | Drag-drop is nice |
| Ingesting many files at once | **CLI** | Terminal stays open; UI can lose progress if tab is closed |
| Scripting or automating | **CLI** | Obvious |
| Running lint with `--deep` (contradictions check) | **CLI** | Deep lint is CLI-only |
| Quick "is this source ingested?" check | **CLI** | `wiki sources list` is faster than clicking |

**Rule of thumb:** use the web UI when you're in "browsing mode" and the CLI when you're in "batch mode".

---

## Common Workflows

### Adding and ingesting a document

You have two options. Both end up with the document processed and filed into your wiki.

#### Option A — Web UI (recommended for one-off files)

1. Open `http://127.0.0.1:8000/ingest`
2. Drag your file into the drop zone (or click to browse)
3. The file appears under **Pending Sources**
4. Click **Ingest →** next to the file
5. Watch the live log as Qwen3 reads, extracts, and writes wiki pages

**Note:** if you close the tab or navigate away, the live log disappears. The ingest itself keeps running in the background though — check `wiki sources list` or go back to `/sources` to see the final status.

#### Option B — CLI (recommended for many files)

```bash
# Copy all PDFs from a folder into raw/ and register them
wiki add ~/Downloads/research-papers/ -r

# Check what's pending
wiki sources list --status pending

# Ingest everything that's pending, one by one, with live console output
wiki ingest --batch
```

Expect **2-15 minutes per document**, depending on type and size. PDFs and DOCX are slower than markdown.

---

### Asking a question

#### Web UI

1. Open `http://127.0.0.1:8000/query`
2. **Pick a scope** (see [Query Scope](#query-scope--wiki-vs-raw-vs-hybrid) below):
   - 📖 **Wiki** — for thematic questions
   - 📥 **Raw** — for looking up specific facts, dates, names
   - 🔀 **Hybrid** — when unsure
3. Type your question, hit Enter
4. Watch as LLM-Wiki classifies your intent, retrieves pages, and streams an answer
5. If the answer is worth keeping, type a slug in the **Save as synthesis** box and click the button — it becomes a new page in your wiki

#### CLI

```bash
# Default — wiki scope with intent classification
wiki query "what are the main themes across my sources?"

# Look up a specific fact in the original documents
wiki query "when is the capstone final report due" --scope raw

# Save the answer as a new wiki page
wiki query "compare RAG and fine-tuning" --scope hybrid --save-as rag-vs-fine-tuning

# Skip intent classification to save ~3 seconds
wiki query "summarize my sources" --no-intent-classify
```

---

### Cleaning up: deleting a source

This removes a source from tracking and deletes the file from `raw/`. **It does NOT delete wiki pages that were created from this source** — those stay as "orphaned knowledge" (run `wiki lint --fix` afterward to clean up dangling references).

#### Web UI

1. Open `/sources`
2. Click the source row to open its detail page
3. Click **🗑 Delete** (top-right of the page)
4. Confirm the dialog

#### CLI

```bash
wiki sources rm 3          # source ID 3
wiki sources rm 3 --keep-file   # unlink from DB but leave the file in raw/
```

---

### Re-ingesting a source

Useful when:
- You blocked a previous ingest mid-flight (Ctrl+C'd or closed the tab)
- The extraction produced poor results and you want to retry
- You've edited the source file and want new pages written

Re-ingesting **merges into existing wiki pages** rather than duplicating them.

#### Web UI

1. Open `/sources`
2. Click the source row to open its detail page
3. Click **↻ Re-ingest** (top-right, blue button)
4. You'll be redirected to `/ingest` where the source now appears in the pending list
5. Click **Ingest →** to run the pipeline

#### CLI

There's no dedicated re-ingest command in the CLI yet, but you can achieve the same by:

```bash
# Mark it as pending via SQL (temporary workaround)
sqlite3 .wiki/state.sqlite "UPDATE sources SET status='pending' WHERE id=3;"

# Then run ingest on that one source
wiki ingest 3
```

---

### Browsing the wiki visually

Three ways to explore what's in your wiki:

#### 1. The web UI's graph view

Open `http://127.0.0.1:8000/graph`. You'll see all pages as nodes, color-coded:

- 🟣 **Purple** — sources (your original documents)
- 🟠 **Orange** — entities (people, organizations, products)
- 🟢 **Green** — concepts (ideas, techniques, theories)
- 🌸 **Pink** — synthesis pages (saved answers)

Hover over a node to see its connection count. Click to open the page. Use the search box to filter.

#### 2. Obsidian (recommended for deep reading)

Open the `wiki/` subfolder as a vault. Obsidian gives you:
- Live markdown preview
- Backlink panel showing which other pages link here
- Graph view with filters
- Search across all content

#### 3. Terminal

```bash
# List all pages
ls wiki/sources/ wiki/entities/ wiki/concepts/ wiki/synthesis/

# Read a specific page
cat wiki/concepts/retrieval-augmented-generation.md

# Find pages mentioning a term
grep -r "attention" wiki/
```

---

### Keeping the wiki healthy

As the wiki grows, issues accumulate: broken wikilinks, orphaned pages, missing frontmatter, stale claims. Lint periodically.

```bash
# Quick health check
wiki lint

# Auto-fix what can be fixed mechanically (broken links, missing frontmatter)
wiki lint --fix

# Deep check — also uses Qwen to look for contradictions across pages
# (slow, ~5-10 min, CLI-only)
wiki lint --deep
```

The web UI at `/lint` shows the same report but without `--deep`.

---

## Query Scope — Wiki vs Raw vs Hybrid

This is the single most important concept for getting good answers. The wiki has two layers of content:

1. **Wiki pages** (`wiki/`) — the LLM's *summaries* of your sources. Concise, thematic, cross-referenced. Ideal for understanding ideas.
2. **Raw documents** (`raw/`) — the *original* source files. Verbose, with every detail preserved. Ideal for exact lookups.

Different questions work better against different layers.

| Scope | What it searches | Best for | Example question |
|---|---|---|---|
| 📖 **Wiki** | LLM summaries in `wiki/` | Thematic, conceptual, "big picture" questions | *"How does RAG relate to transformers?"* |
| 📥 **Raw** | Original documents in `raw/` | Specific facts, dates, exact quotes, numbers | *"When is the final project due?"* |
| 🔀 **Hybrid** | Both, results merged | When you want both specific details AND conceptual context | *"What are the key dates in my AIML course and what topics do they cover?"* |

**Rule of thumb:** if your question has a specific date, name, number, or line-item in it, use **Raw**. Otherwise start with **Wiki** and switch to **Hybrid** if the answer lacks detail.

---

## Full CLI Command Reference

All commands assume you've activated the venv (`source .venv/bin/activate`). Run `wiki --help` at any time for the current list.

### Setup & info

```bash
wiki version                    # Show version
wiki status                     # Show project stats: sources, pages, config
wiki init ~/path/to/new-wiki    # Initialize a new wiki in a given folder
```

### Managing sources (raw documents)

```bash
# Add files to the raw collection
wiki add file.pdf                       # Add a single file
wiki add ~/Downloads/docs/ -r           # Add a folder recursively
wiki add *.md                           # Add with glob

# List and inspect
wiki sources list                       # All sources
wiki sources list --status pending      # Only unprocessed
wiki sources list --status ingested     # Only processed
wiki sources show 3                     # Details for source ID 3

# Remove
wiki sources rm 3                       # Remove and delete file from raw/
wiki sources rm 3 --keep-file           # Remove from DB only
```

### Ingesting into the wiki

```bash
wiki ingest                             # Interactive — prompts for each pending source
wiki ingest --batch                     # Non-interactive — processes all pending sources
wiki ingest 3                           # Ingest a specific source by ID
wiki ingest --dry-run                   # Show what would be done without doing it
```

### Querying the wiki

```bash
wiki query "your question here"         # Default: wiki scope, hybrid mode, with intent classification

# Scope flags (which collection to search)
wiki query "..." --scope wiki           # LLM-summarized pages (default)
wiki query "..." --scope raw            # Original documents
wiki query "..." --scope hybrid         # Both

# Mode flags (how to search)
wiki query "..." --lex                  # BM25 keyword only, fastest
wiki query "..." --vec                  # Vector semantic only
wiki query "..." --mode hybrid          # Both + LLM rerank (default, best quality)

# Other flags
wiki query "..." -n 10                  # Return top 10 hits instead of 8
wiki query "..." --min-score 0.3        # Drop low-confidence hits
wiki query "..." --no-rerank            # Skip LLM reranking (faster)
wiki query "..." --no-intent-classify   # Skip intent classification (~3 sec faster)
wiki query "..." --save-as my-answer    # Save as wiki/synthesis/my-answer.md
```

### Keeping things healthy

```bash
wiki lint                               # Quick health report
wiki lint --fix                         # Auto-fix mechanical issues
wiki lint --deep                        # Add LLM contradiction detection (slow)

wiki reindex                            # Force rebuild of the search index
                                        # Use after manually editing wiki pages
```

### Running the web UI

```bash
wiki serve                              # Start on http://127.0.0.1:8000
wiki serve --port 8080                  # Custom port
wiki serve --no-browser                 # Don't auto-open browser
```

---

## Troubleshooting

### `zsh: command not found: wiki`

Your venv isn't activated. The prompt should start with `(LLM-Wiki)` when it is.

```bash
cd ~/Documents/LLM-Wiki
source .venv/bin/activate
```

### `Ollama not ready` or `connection refused`

Ollama isn't running. Start the Ollama app or run `ollama serve &`.

### `Model qwen3:14b not found`

Pull the model first (one-time, ~9GB download):

```bash
ollama pull qwen3:14b
```

### Ingest UI shows "Ingesting…" forever / progress vanished

The SSE stream is tied to the browser tab. If you navigate away or close the tab, the progress display is lost — **but the ingest is still running on the server**. To check actual status:

```bash
wiki sources list           # See if status flipped to "ingested"
```

If it's still `pending` after a long time (>20 min for a PDF), the ingest genuinely hung. Restart the server:

```bash
# Ctrl+C the wiki serve terminal
wiki serve
```

### Query answers "the sources don't contain this info" for a fact I know is in a document

You're probably using **Wiki** scope when you should use **Raw**. The wiki layer contains *summaries*, not full text — specific dates, names, and line-items often get compressed away. Switch scope to **Raw** (web UI toggle or `--scope raw` CLI flag).

### Web UI shows "Internal Server Error"

Check the terminal where `wiki serve` is running — there'll be a Python traceback. Common causes:
- `ingest_raw.py` or `ingest_llm.py` got corrupted (run: `wc -l src/llm_wiki/ingest_*.py` — should show `341` and `836`)
- Ollama stopped midway through a request
- QMD binary isn't installed

### Graph view is empty or shows fewer nodes than expected

```bash
wiki reindex      # Rebuilds the search index
wiki lint --fix   # Fixes orphaned and broken-wikilink pages
```

### `Operation not permitted` errors from `cat`, `head`, `wc`

macOS Full Disk Access permission issue. Open **System Settings** → **Privacy & Security** → **Full Disk Access**, add (or enable) your Terminal app, then fully quit and reopen Terminal (Cmd+Q).

---

## First-Time Setup (If You're Starting Fresh)

Most users won't need this — the project is already set up. Include for completeness.

### Prerequisites

- **macOS** (tested on M3; should work on Intel Macs and Linux too)
- **Python 3.11+** (the project is built against 3.13)
- **Node.js 22+** (needed for QMD, the search engine)
- **Homebrew SQLite** — `brew install sqlite` (for QMD's vector extensions)
- **~15GB free disk space** (for Qwen3-14B and QMD's support models)

### Install

```bash
# 1. Clone the repo
git clone https://github.com/NiharShrotri/llm-wiki.git
cd llm-wiki

# 2. Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Install QMD (the search backend)
npm install -g @tobilu/qmd

# 4. Install Ollama and pull the model
# Get Ollama from https://ollama.com (or brew install --cask ollama)
ollama pull qwen3:14b

# 5. Initialize the wiki structure
wiki init .

# 6. Start serving
wiki serve
```

Open `http://127.0.0.1:8000` and start adding documents.

---

## Getting Help

- **Project README** (architecture, internals, design decisions) — [README.md](./README.md)
- **GitHub Issues** — https://github.com/NiharShrotri/llm-wiki/issues
- **Karpathy's original LLM-Wiki gist** (the pattern this implements) — https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

When reporting bugs, please include:
- Output of `wiki version` and `wiki status`
- The command you ran and the full error message
- Output of `wc -l src/llm_wiki/*.py src/llm_wiki/webapp/**/*.py` (helps spot file corruption)

---

*Last updated: LLM-Wiki v0.7.0*
