# LaceMem-1.0

> **Current branch:** `main` — default **raw** ingest (standard `RawMemoryManager` / one-shot `raw_memory_insert`). For **pending topic & time** across turns, use branch **`context-aware`** ([Git branches](#git-branches)).

## Overview

**LaceMem** (Layered Architecture for Conversational Evidence Memory) is a coarse-to-fine memory hierarchy of three layers — **Index** (atomic semantic tuples for search), **Edge** (graph links for multi-hop expansion), and **Raw** (verbatim dialogue for grounding).

This repository (**LaceMem-1.0**) materialises those layers in PostgreSQL: it ingests [LoCoMo](https://github.com/snap-research/locomo)-style multi-session dialogue, builds Raw → Index → Edge via LLM-backed managers, and can export the same tables to SQLite for tools that expect a single-file database.

## Method summary

**LaceMem** organises dialogue into a coarse-to-fine three-layer hierarchy: an *Index layer* of atomic semantic tuples for fine-grained search, an *Edge layer* of graph links for multi-hop expansion, and a *Raw layer* of verbatim dialogue for grounded generation.

## Git branches

| Branch | Raw-memory behaviour | Everything else |
|--------|----------------------|-----------------|
| **`main`** | Standard prompt: decide whether to call `raw_memory_insert`; one-shot tool flow. | Baseline `run_eval.py`, `IndexManager`, `LinkManager`, schema, Postgres defaults (`LaceMem`). |
| **`context-aware`** | **Context-aware ingest:** keeps in-memory **pending topic / time** across turns, exposes `pending_*` tools, and uses an expanded `prompts/raw_memory_manager.py` so short or coreferential utterances can be rewritten into standalone text before storage. | Same DB name (**`LaceMem`**), same indexing/edge pipeline and `run_eval.py` layout as `main`; differs mainly in `RawMemoryManager` + raw prompt. |

Check out a branch to run that code path; database setup (`createdb "LaceMem"`, `psql … schema.sql`) is identical.

## What is in this repository


| Path                                              | Role                                                                                     |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `db/schema.sql`                                   | DDL for `raw_memory`, `memory_index`, `memory_index_edge`                                |
| `db/db_conn.py`                                   | PostgreSQL connection (`MEM_EVAL_DB_URL`, default `postgresql://localhost:5432/LaceMem`) |
| `db/db_raw.py`, `db/db_index.py`, `db/db_edge.py` | Typed accessors for each layer                                                           |
| `prompts/`                                        | LLM prompts for raw ingest, indexing, and linking                                        |
| `db_managers/db_managers.py`                      | `RawMemoryManager`, `IndexManager`, `LinkManager` orchestration                          |
| `men_llm/llm_client.py`                           | OpenAI client wrapper (`OPENAI_API_KEY` or `./api.key`)                                  |
| `run_eval.py`                                     | End-to-end ingest of one LoCoMo sample from `data/locomo10.json` into Postgres           |
| `pg_to_sqlite.py`                                 | Copy the three LaceMem tables from Postgres into `LaceMem.sqlite.db`                     |
| `requirements.txt`                                | Python dependencies (`pip install -r requirements.txt`)                                  |
| `data/locomo10.json`                              | LoCoMo-10 dataset (place or symlink here)                                                |


## Setup

### 1. Python environment

Use Python 3.10+ (3.11 recommended). Create a virtual environment in the project root, activate it, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`psycopg[binary]` powers `run_eval` / `db/db_conn.py`; `psycopg2-binary` is only needed for `pg_to_sqlite.py`.

### 2. PostgreSQL

Install and start PostgreSQL locally (or point `MEM_EVAL_DB_URL` at a remote instance). Ensure `psql` and `createdb` / `dropdb` are on your `PATH`.

### 3. OpenAI API key

Either export:

```bash
export OPENAI_API_KEY=sk-...
```

or create a file `api.key` in the project root containing the key on a single line (as expected by `LLMClient` / `RawMemoryManager`).

Optional: `MEM_EVAL_MODEL` selects the chat model (default `gpt-4o-mini`).

### 4. LoCoMo data

Place `locomo10.json` at `**data/locomo10.json**` (or change `DATA_PATH` in `run_eval.py`).

## Usage

### Create the database and tables

```bash
dropdb "LaceMem"   # optional; remove old DB if it exists and nothing else needs it
createdb "LaceMem"
psql -d "LaceMem" -f db/schema.sql -q
```

If your Postgres role or host differ from the default, set `MEM_EVAL_DB_URL`, e.g.:

```bash
export MEM_EVAL_DB_URL=postgresql://user:password@localhost:5432/LaceMem
```

### Ingest one conversation (LaceMem build)

1. Set `target_sample_id` in `run_eval.py` to the LoCoMo sample you want (e.g. `conv-26`, `conv-30`).
2. Run:

```bash
python run_eval.py
```

This fills **Raw** turns, then **Index**, then **Edge** links for the selected sample.

### Export Postgres → SQLite (optional)

For downstream stacks that expect SQLite:

```bash
python pg_to_sqlite.py
```

Writes `**LaceMem.sqlite.db**` with tables `raw_memory`, `memory_index`, `memory_index_edge`. Connection parameters follow `PG*` env vars and defaults in `pg_to_sqlite.py`.

## Notes

- `**run_eval.py**` currently targets a single `sample_id`; adjust `target_sample_id` per run.
- Ensure no other process holds open connections to `"LaceMem"` before `dropdb` (terminate sessions or use `DROP DATABASE ... WITH (FORCE)` in PostgreSQL 13+).
- `**.ruff_cache/**` is created by the [Ruff](https://docs.astral.sh/ruff/) linter when you run `ruff check`; it is safe to delete and will be recreated.

## Acknowledgements

- [LoCoMo](https://github.com/snap-research/locomo) — long-term conversational memory benchmark and data format.

