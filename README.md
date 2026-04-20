# LLM-Wiki

> A local, LLM-maintained personal knowledge base. Drop documents in, watch an LLM compile them into a living, interlinked Obsidian wiki you can search and query.

Feel free to fork and don't forget to give it a Star ⭐️ for better reach!

-------------------------
Hello, I'm Nihar Shrotri, working as an AI Consultant.
I'm currently pursuing my PhD in Artificial Intelligence and Machine Learning

Let's connect on LinkedIn for a Chat: https://www.linkedin.com/in/niharshrotri/
-------------------------

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/LLM-Qwen3--14B-purple.svg)](https://ollama.com/library/qwen3)
[![Local-first](https://img.shields.io/badge/runs-100%25_local-green.svg)](#)

Built on the pattern Andrej Karpathy described in his [LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): instead of retrieving from raw documents at query time (classic RAG), an LLM incrementally **compiles** your sources into a structured, cross-linked markdown wiki that sits between you and the raw documents. The wiki is a **persistent, compounding artifact** — the cross-references are already there, the contradictions have already been flagged, the synthesis already reflects everything you've read.

You never write the wiki yourself. The LLM does all the grunt work: summarizing, cross-referencing, filing, bookkeeping. You bring the sources and ask the questions.

Runs 100% locally on Apple Silicon or anywhere Ollama works. No API keys, no cloud, no data leaving your machine.

## What it does

```bash
# Drop files in (PDFs, markdown, HTML, DOCX, text)
wiki add ~/Documents/papers --recursive

# Watch Qwen3 read them and build an interlinked wiki
wiki ingest

# Ask questions — it searches the compiled wiki and cites its sources
wiki query "what's the main argument about X?"

# Health-check the knowledge base
wiki lint --fix

# Browse the whole thing in Obsidian (graph view, backlinks, everything)
open wiki/
```

Every ingest produces a cluster of `sources/`, `entities/`, and `concepts/` pages with YAML frontmatter and `[[wikilinks]]` between them. Every query pulls the top-ranked pages via hybrid BM25 + vector + LLM-rerank search, then synthesizes a cited answer. Every lint run catches broken links, orphan pages, malformed frontmatter, and (optionally, using the LLM) contradictions between pages.

## Features

### Core capabilities
- **Incremental ingest** — drop a file, run `wiki ingest`, get 8–15 cross-linked wiki pages
- **Structured extraction** — Qwen3 identifies entities (people, orgs, models), concepts, and key takeaways per source
- **Smart merging** — re-ingesting related sources updates existing entity/concept pages instead of overwriting them, preserving provenance
- **Hybrid search** — BM25 full-text + vector embeddings + LLM reranking (all local, via [QMD](https://github.com/tobi/qmd))
- **3-way query scope** — `Wiki` (thematic answers from LLM-compiled pages), `Raw` (exact lookups in original documents), or `Hybrid` (both)
- **Intent classification** — casual messages ("hi", "thanks") skip retrieval and get a quick reply, saving ~30 seconds per chitchat turn
- **Cited synthesis** — queries return markdown answers with `[[wikilinks]]` pointing to the pages that support each claim
- **Write-back** — save good answers as new `synthesis/` pages with `--save-as`, so your explorations compound in the knowledge base
- **Wiki linting** — automated health checks for broken links, orphans, malformed frontmatter, noise in sources, and (with `--deep`) LLM-powered contradiction detection between pages
- **Auto-fix** — most stylistic issues resolve with one command
- **Auto-reindex** — search index refreshes automatically after ingest and lint; new pages are queryable immediately

### Web UI
A full web interface at `http://127.0.0.1:8000` after `wiki serve`:
- **Dashboard** — project stats and recent activity
- **Sources** — list, inspect, delete, or re-ingest sources with one click
- **Ingest** — drag-and-drop upload, live progress log, persistent jobs that **survive tab close and server restart**
- **Jobs** — history of all ingest runs with live progress bars and error details
- **Query** — chat-style interface with streaming synthesis, scope toggle, save-as-synthesis button
- **Lint** — interactive lint report with one-click auto-fix
- **Graph** — D3 force-directed visualization of the full wiki, color-coded by page type

### Supported input formats
`.pdf` · `.md` · `.html` · `.docx` · `.txt`

### Obsidian integration
The `wiki/` folder is a ready-made Obsidian vault with:
- Color-coded graph view (sources, entities, concepts, synthesis each get their own color)
- YAML frontmatter compatible with the Dataview plugin
- All cross-references as native `[[wikilinks]]` so backlinks, outgoing-links, and graph traversal all work

## Architecture

Three layers, per Karpathy:

```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   raw/        │ → │   LLM Agent   │ → │   wiki/       │
│ Your docs     │   │  (Qwen3-14B)  │   │ Markdown,     │
│ (immutable)   │   │               │   │ auto-linked   │
└───────────────┘   └───────────────┘   └───────────────┘
                          │                     │
                          ▼                     ▼
                    ┌───────────────┐   ┌───────────────┐
                    │ schema/       │   │   Obsidian    │
                    │ AGENTS.md     │   │  graph view   │
                    │ (the rules)   │   │  + editing    │
                    └───────────────┘   └───────────────┘
```

- **`raw/`** — your source documents. Immutable. The agent reads but never modifies.
- **`wiki/`** — LLM-maintained markdown. One folder per page type (`sources/`, `entities/`, `concepts/`, `synthesis/`) plus auto-generated `index.md` and `log.md`. Open this in Obsidian.
- **`schema/AGENTS.md`** — the conventions file. Tells the LLM how to format pages, when to merge vs create, how to cite, how to handle contradictions. Edit as your preferences evolve.
- **`.wiki/`** — internal state: SQLite ingest history, QMD search index, config. Git-ignored.

### The ingest pipeline

Each source goes through three LLM passes:

1. **Extraction** (thinking mode on) — Qwen3 reads the source and returns structured JSON: summary, key takeaways, named entities, concepts, tags.
2. **Page drafting** (streaming, thinking mode off) — one call per entity/concept. Draft a new page from scratch, or *merge* new information into an existing page (preserving prior content, updating dates, appending to `sources:` frontmatter).
3. **Source summary** — write the `sources/<slug>.md` page listing every wiki page touched by this source for provenance.

After the three passes: `index.md` is rebuilt, `log.md` is appended, and QMD's search index is updated automatically.

### The query pipeline

1. **Hybrid search** via QMD — BM25 full-text + vector similarity + LLM reranker, all local
2. **Top-K page hydration** — load full content of the top 5–8 hits
3. **Synthesis** — Qwen3 writes a cited markdown answer using `[[wikilinks]]` to reference the pages
4. **(Optional) save-back** — `--save-as` files the answer as a new `synthesis/` page

## Stack

| Layer | Component | Why |
|---|---|---|
| LLM | [Ollama](https://ollama.com) + [Qwen3-14B](https://ollama.com/library/qwen3:14b) Q4_K_M | Strong reasoning, 40K context, thinking mode, 9.3GB on disk |
| Search | [QMD](https://github.com/tobi/qmd) (BM25 + vector + rerank) | All local, SQLite-backed, handles the heavy lifting |
| Embeddings | EmbeddingGemma-300M (via QMD) | Small footprint, high quality |
| Reranker | Qwen3-Reranker-0.6B (via QMD) | Fast cross-encoder rerank |
| CLI | [Typer](https://typer.tiangolo.com) + [Rich](https://rich.readthedocs.io) | Great UX, colored output, progress bars |
| Parsers | pypdf, python-docx, beautifulsoup4, lxml | Cover the main document formats |
| Vault | [Obsidian](https://obsidian.md) | Best-in-class graph view and backlink UX — you don't have to build it |

No cloud services. No API keys. No data leaves your machine.

## Requirements

- **Python 3.11+**
- **Node.js 18+** (for QMD)
- **Ollama** with the `qwen3:14b` model pulled (~9.3GB)
- **QMD** (`npm install -g @tobilu/qmd`)
- **Homebrew SQLite** on macOS (`brew install sqlite`)
- **~15GB free disk space** for models and embeddings
- **~12GB RAM** recommended (16GB+ for comfort)
- **Obsidian** (optional but strongly recommended for browsing)

Tested on macOS (Apple Silicon, M3 Pro 18GB). Should work on Linux; Windows untested.

## Installation

```bash
# Clone
git clone https://github.com/YOUR-USERNAME/llm-wiki.git
cd llm-wiki

# Create a virtual environment (uv is faster than pip, either works)
uv venv
source .venv/bin/activate
uv pip install -e .

# Pull the LLM (one-time, ~9.3GB)
ollama pull qwen3:14b

# Install QMD (the search backend)
npm install -g @tobilu/qmd

# Verify
wiki version
wiki --help
```

## Quick start

```bash
# 1. Create a wiki in a folder of your choosing
mkdir my-wiki && cd my-wiki
wiki init

# 2. Drop some source documents in raw/, or use:
wiki add ~/Documents/papers --recursive

# 3. Run ingest (interactive by default — shows you entities/concepts
#    before filing, with a y/n prompt per source)
wiki ingest

# First query triggers QMD to download its embedding + reranker models
# (~2GB, one-time). Subsequent queries are fast.

# 4. Ask questions
wiki query "what are the main themes across these documents?"

# 5. Save a good answer as a synthesis page
wiki query "compare X vs Y" --save-as x-vs-y-comparison

# 6. Health-check and auto-fix
wiki lint --fix

# 7. Browse the vault in Obsidian
open wiki/   # then "Open folder as vault"
```

## Commands

| Command | Purpose |
|---|---|
| `wiki init [path]` | Scaffold a new wiki project |
| `wiki add <file-or-folder> [-r]` | Copy sources into `raw/` and register for ingest |
| `wiki sources list` | List all tracked sources with status |
| `wiki sources show <id>` | Show metadata + text preview for one source |
| `wiki sources rm <id>` | Remove a source from tracking |
| `wiki ingest [source_id]` | Run the 3-pass LLM ingest pipeline |
| `wiki query "<question>" [--scope wiki\|raw\|hybrid] [--save-as <slug>]` | Search + synthesize a cited answer |
| `wiki reindex` | Force rebuild of the QMD search index |
| `wiki lint [--deep] [--fix]` | Health-check the wiki |
| `wiki status` | Show project stats, paths, config, backend health |
| `wiki serve [--port N]` | Launch the web UI at `http://127.0.0.1:8000` |

Run `wiki <command> --help` for full options on any command. See [USAGE.md](./USAGE.md) for a full walkthrough.

## Example output

A real ingest against `notes.txt` (28 words about Qwen3):

```
Source #1  raw/notes.txt
  parsing…
  extracting entities and concepts (thinking mode)…

Title: Quick Notes on Qwen
Slug:  quick-notes-on-qwen

Summary:
  Qwen is a family of large language models developed by Alibaba Cloud.
  The latest version, Qwen3, introduces a thinking mode designed to enhance
  performance on complex reasoning tasks.

Entities (3):
  + alibaba-cloud (organization)  Alibaba Cloud
  + qwen (product)                Qwen
  + qwen3 (product)               Qwen3

Concepts (2):
  + large-language-models                 Large Language Models
  + thinking-mode-for-complex-reasoning   Thinking Mode for Complex Reasoning

File these? Will create/update ~6 wiki pages. [Y/n]: Y

created entity alibaba-cloud
created entity qwen
created entity qwen3
created concept large-language-models
created concept thinking-mode-for-complex-reasoning
created source  quick-notes-on-qwen

✓ Ingested Quick Notes on Qwen — 6 created, 0 updated
```

That's **6 cross-linked pages from a 28-word input**, each with YAML frontmatter, `[[wikilinks]]` between them, and provenance back to the source. Open Obsidian's graph view and you'll see the cluster light up.

A real query against 11 ingested pages:

```
> wiki query "how does multi-head attention differ from self-attention?"

  searching wiki (BM25 + vector + rerank)…
  found 8 relevant page(s):
    1. 0.93 concepts/multi-head-attention.md      Multi-Head Attention
    2. 0.55 concepts/self-attention-mechanism.md  Self-Attention Mechanism
    3. 0.40 entities/attention-is-all-you-need.md Attention is All You Need
    ...

  synthesizing answer…

Multi-head attention and self-attention are related but distinct mechanisms:

1. **Scope and Parallelism**
   - Self-attention is a single mechanism where each position in the input
     computes attention weights based on all other positions
     [[concepts/self-attention-mechanism]].
   - Multi-head attention extends this by using multiple parallel attention
     heads, allowing the model to focus on diverse patterns simultaneously
     [[concepts/multi-head-attention]].

2. **Information Capture**
   - Self-attention focuses on a single representation.
   - Multi-head aggregates information from multiple heads, each capturing
     different aspects (syntactic vs semantic, etc.)
     [[concepts/multi-head-attention]].

[... etc.]
```

Every claim is cited. Every citation points to a page that actually exists.

## Lint example

```
> wiki lint

╭─────────── Lint Report ────────────╮
│ Health score: 57/100               │
│ Pages checked: 12                  │
│                                    │
│   2 errors · 21 warnings · 0 infos │
╰────────────────────────────────────╯

──── Errors (2) ────

  synthesis/transformers-and-llms.md
    ✗ broken_wikilink: Broken wikilink: [[entities/introduction-to-transformers]]
      → Either create the page or remove the link.

──── Warnings (21) ────

  entities/qwen.md
    ! malformed_wikilink: 'sources/quick-notes-on-qwen.md' should be
      'sources/quick-notes-on-qwen'
      ✓ auto-fixable

  [... 20 more warnings ...]

> wiki lint --fix
✓ auto-fixed: 11

> wiki lint
╭─────────── Lint Report ───────────╮
│ Health score: 100/100             │
│   0 errors · 0 warnings · 0 infos │
╰───────────────────────────────────╯
✓ No issues found. Your wiki is in good shape!
```

## Project status

**Current version: v0.8.1** — production-ready for personal use.

| Stage | Scope | Status |
|---|---|---|
| 1 | Scaffolding, CLI, Obsidian vault config | ✅ Done |
| 2 | Parsers (PDF, MD, HTML, DOCX, TXT), dedupe, `wiki add` | ✅ Done |
| 3 | LLM ingest pipeline (3 passes, streaming, merge-path) | ✅ Done |
| 4 | QMD search + `wiki query` with citation + save-back | ✅ Done |
| 5 | Lint checks + auto-fix + deep contradiction detection | ✅ Done |
| 6 | FastAPI + HTMX web UI (7 pages: Dashboard, Sources, Ingest, Jobs, Query, Lint, Graph) | ✅ Done |
| 7 (v0.7.0) | Source CRUD, intent classification, 3-way scope toggle | ✅ Done |
| 8 (v0.8.0) | Persistent ingest jobs (survive tab close, server restart) | ✅ Done |
| 8.1 | Auto-reindex after ingest and lint | ✅ Done |

### Possible future work
- Hugging Face Spaces deployment (smaller model, API-compatible)
- Dashboard showing live active-job count
- Static HTML export for sharing the wiki
- Multi-user / team features
- Mobile-friendly web UI
- Fine-tuned query expansion model
- Confidence scoring per extracted claim
- OCR support for scanned PDFs
- EPUB support

## Credits

- **[Andrej Karpathy](https://karpathy.ai/)** — for the LLM-Wiki pattern described in [this gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). This project is a direct implementation of the idea.
- **[QMD](https://github.com/tobi/qmd)** by Tobi Lütke — the hybrid search backend that does all the heavy lifting for query-time retrieval.
- **[Qwen3](https://qwenlm.github.io/blog/qwen3/)** by Alibaba Cloud — the local LLM doing the reading, writing, and synthesis.
- **[Ollama](https://ollama.com)** — the runtime that makes local LLM inference painless on Apple Silicon.
- **[Obsidian](https://obsidian.md)** — saved me from writing my own graph view.

## License

MIT — see [LICENSE](LICENSE).

---

*"The tedious part of maintaining a knowledge base is not the reading or the thinking — it's the bookkeeping. Humans abandon wikis because the maintenance burden grows faster than the value. LLMs don't get bored, don't forget to update a cross-reference, and can touch 15 files in one pass."*
— Karpathy
