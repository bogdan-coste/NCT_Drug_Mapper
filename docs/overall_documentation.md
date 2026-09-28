# Therapeutic Category Pipeline — Overall Documentation

A system that takes a drug term from a clinical trial and decides where it belongs in a
biomedical ontology, using ClinicalTrials.gov evidence, an LLM, and SapBERT + Qdrant retrieval.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Design Philosophy](#2-design-philosophy)
3. [Pipeline Flow](#3-pipeline-flow)
4. [Components](#4-components)
5. [Input Data: `SCRIBE_terms.xlsx`](#5-input-data-scribe_termsxlsx)
6. [Output Format](#6-output-format)
7. [Configuration](#7-configuration)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Current Gap: Excel Is Not Yet Wired In](#9-current-gap-excel-is-not-yet-wired-in)

---

## 1. The Problem

Ontology curators receive a steady stream of new drug terms from clinical trials:

- **Sponsor codes** with no semantic content — `PEP08`, `ZL-1310`, `CR-001`, `MZ-1866`
- **Combination regimens** — `NALIRIFOX`, `HRS-4642+AG+Nimotuzumab`
- **Cell therapies** — `EB-SELECT Dual-Target CAR-NK Cells`
- **Brand names** — `CLIGAVYX`, `VYBRIQUE`

For each one, a curator must answer a single question: **what is the correct parent class
in the ontology?** Today that answer is produced by hand — reading the trial record,
inferring the drug's modality and mechanism, and picking a superclass. `SCRIBE_terms.xlsx`
is a snapshot of exactly that manual work.

This project automates the decision and produces a curator-reviewable answer with a
justification and a confidence level.

---

## 2. Design Philosophy

Three principles drive the whole codebase. Most of them are enforced in
`src/prompts/prompt_templates.py`, which is where the real domain logic lives.

### 2.1 Strict grounding — never infer from the name

The model may only assert what is explicitly present in the supplied context. The prompts
deliberately forbid the shortcuts an LLM would otherwise take:

| Forbidden shortcut | Why it is wrong |
|---|---|
| `-mab` suffix → monoclonal antibody | Naming conventions are not evidence |
| Intravenous route → biologic | Route of administration is not a modality |
| Cancer indication → targeted therapy | An indication is not a mechanism |
| Name spelling → drug class | Sponsor codes carry no meaning |

`targeted therapy` may only be selected when the definition explicitly names a molecular,
antigenic, cellular, or pathway target.

### 2.2 Prefer a correct general answer over a guessed specific one

When the evidence does not support a specific modality, the pipeline climbs to a broader
supported parent — `drug`, `investigational drug`, `antineoplastic drug`, `biologic therapy`,
`cell therapy`. Two explicit escape hatches exist:

- **`I don't know`** — Step 1, when the context says nothing reliable about the term
- **`NONE_OF_THE_ABOVE`** — Step 3, when no retrieved candidate is semantically appropriate

For a curator, a vague answer costs a minute of review. A confidently wrong answer
corrupts the ontology.

### 2.3 Search by what the drug *is*, not by what it is *called*

This is the central, non-obvious idea.

`PEP08` has no useful semantic neighbours in an embedding space — it is an arbitrary string.
So the pipeline **first derives what the drug is**, then searches the ontology using that
concept instead of the raw name:

```
"PEP08"  ──►  "MTA-cooperative PRMT5 inhibitor"  ──►  enzyme_inhibitor_therapy
 (opaque)          (derived from trial evidence)         (found in ontology)
```

SapBERT is trained on biomedical language, not on sponsor codes. Giving it a class name
instead of a product code is the difference between a meaningful search and noise.

### 2.4 The LLM decides; the embedding only nominates

Retrieval produces candidates, not answers. Qdrant's ranking is explicitly **not** treated
as the result — Step 3's prompt states *"do not select a candidate based on embedding
similarity alone."* The final choice is a reasoned selection over the candidate set,
grounded in the definition.

---

## 3. Pipeline Flow

```
                    Term + NCT ID(s)
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  NCTLoader.fetch_study                   │
        │  GET clinicaltrials.gov/api/v2/studies   │
        └──────────────────┬───────────────────────┘
                           │  raw JSON
                           ▼
        ┌──────────────────────────────────────────┐
        │  NCTDrugContext.from_api                 │
        │  Keeps only drug-relevant fields.        │
        │  Discards dates, sponsor, eligibility,   │
        │  outcomes — trial metadata is noise.     │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  _build_context(trial, term)             │
        │  Assembles the context string.           │
        │  Intervention descriptions are filtered  │
        │  to those mentioning the target term.    │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  STEP 1 — DefinitionGenerator (LLM)      │
        │                                          │
        │  "Subclass of <parent>. <one sentence>.  │
        │   (ClinicalTrials.gov; NCT01234567)"     │
        └──────────────────┬───────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │  definition carried
                ▼                     │  forward to Step 3
        ┌──────────────────────────┐  │
        │  STEP 2 — Superclass     │  │
        │  Suggester (LLM)         │  │
        │                          │  │
        │  Input: the definition    │  │
        │  ONLY — not raw context. │  │
        │                          │  │
        │  → { specific_label,     │  │
        │      broad_label,        │  │
        │      category_label }    │  │
        └──────────┬───────────────┘  │
                   │                  │
                   ▼                  │
        ┌──────────────────────────┐  │
        │  RETRIEVAL               │  │
        │  For each of the 3       │  │
        │  labels:                 │  │
        │    SapBERT embed         │  │
        │    Qdrant top-20 (cosine)│  │
        │  Then dedup by           │  │
        │  display_name, keeping   │  │
        │  the best score → top 20 │  │
        └──────────┬───────────────┘  │
                   │                  │
                   └────────┬─────────┘
                            ▼
        ┌──────────────────────────────────────────┐
        │  STEP 3 — OntologyMapper (LLM)           │
        │                                          │
        │  Inputs: definition + superclass +       │
        │          20 ontology candidates          │
        │                                          │
        │  → { selected_term, confidence,          │
        │      match_type, reason }                │
        └──────────────────────────────────────────┘
```

### Why the flow is shaped this way

**Step 2 reads only the definition, never the raw context.** This is an intentional
bottleneck. The definition has already passed through the grounding rules, so speculation
that was filtered out in Step 1 cannot re-enter in Step 2.

**Retrieval fans out from specific to general.** All three labels are searched
independently because the ontology may be incomplete. If
`MTA-cooperative PRMT5 inhibitor` does not exist as a class, `enzyme inhibitor` or
`antineoplastic drug` almost certainly does — and the broader hit is still a usable answer.

**`match_type` separates *what was found* from *how sure we are*.** A result of
`broader_available` with `high` confidence is not a weak mapping — it is a signal that the
ontology itself needs a new, more specific class. That distinction is what makes the output
actionable for a curator.

**Fail-fast at every stage.** `TherapeuticCategory.run()` returns a partial result with an
`error` key the moment any step fails, preserving whatever was already produced. A failed
Step 3 still hands back the definition and superclass labels.

---

## 4. Components

| Path | Responsibility |
|---|---|
| `src/ingestion/nct_loader.py` | Fetches a single trial record from ClinicalTrials.gov API v2 |
| `src/schemas/models.py` | `NCTDrugContext` field extraction, plus all configuration classes (`LLMParameters`, `SapBERTParameters`, `QdrantParameters`, `PathParameters`, `ServerParameters`, `AppConfig`) — each with a `from_env()` constructor |
| `src/prompts/prompt_templates.py` | Three system prompts and three `string.Template` user prompts. **All curation logic lives here.** |
| `src/generation/therapeutic_grounded_answer.py` | `DefinitionGenerator`, `SuperclassSuggester`, `OntologyMapper` — one class per step, each binding its own system prompt to its LLM client |
| `src/llm/llm_client.py` | OpenAI Responses API client against the internal proxy; supports plain text (`ask_llm`) and Pydantic-parsed (`ask_llm_pydantic`) responses; sends the `App-Id` header |
| `src/api/sapbert_server.py` | Loads SapBERT locally and serves an OpenAI-compatible `/v1/embeddings` endpoint on port 7997 in a daemon thread. Uses CLS pooling (`last_hidden_state[:, 0, :]`) |
| `src/embedder/sapbert_embedder.py` | OpenAI-style client pointed at the local SapBERT server; `embed_one` / `embed_many` |
| `src/vectordb/vector_store.py` | Low-level Qdrant operations — `create_collection`, `upsert`, `search`. Queries the named vector `sapbert_terminology` with cosine distance |
| `src/retrieval/retriever.py` | `retrieve_for_labels` — multi-label search, deduplication by best score, top-k merge |
| `src/processing/modules/thereapeutic_cathegory_module.py` | The orchestrator: context building, three LLM steps, retrieval, and error handling |
| `main.py` | Interactive REPL — prompts for a term and NCT IDs, prints each stage |
| `src/api/routes.py` | FastAPI service exposing `/run_pipeline`, `/embed`, `/ask_llm`, `/health` |
| `scripts/download_models.py` | One-time download of `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` into `models/sapbert` |
| `scripts/test_therapeutic_pipeline.py` | End-to-end smoke test over a hardcoded list of test cases |

The layout follows the "scalable RAG project" template in `proj_structure_sample.txt`,
keeping only the layers this problem needs. Reranking, caching, BM25/hybrid search, and the
eval harness are deliberately absent.

### Two entry points, one pipeline

`main.py` (REPL) and `src/api/routes.py` (HTTP) wire up identical component graphs. Both
boot the SapBERT server, build the embedder, retriever, and vector store, then construct
`TherapeuticCategory` with three separate `LLMClient` instances — one per step, since each
step needs its own system prompt.

---

## 5. Input Data: `SCRIBE_terms.xlsx`

The workbook is the real-world source of both input and expected output. It has nine sheets;
two of them matter for this pipeline.

### 5.1 `Requested Drug-Trials` — the input queue (50 rows)

Its columns map one-to-one onto `pipeline.run(term, nct_ids)`:

| Excel column | Pipeline usage |
|---|---|
| `Author's Usage` | → `term` |
| `NCT Record ID` | → `nct_ids` (multiple IDs separated by `\|`) |
| `Justification` | Source discriminator: `NCT` vs `KANG:DrugLabel/Guideline` |
| `Proposed New Term` | Curator's manual answer |
| `Comment` | Curator notes (e.g. *"ACTUAL DRUG IS Tb-161 DOTATATE"*) |
| `Curation Date`, `Change made to finding in tool` | Workflow tracking |

Breakdown of the 50 rows:

| Category | Count | Notes |
|---|---|---|
| Rows with NCT IDs | **30** | Directly processable today |
| — of which multi-trial | 6 | Confirms the multi-NCT context merge in `run()` is a real requirement |
| KANG URLs | **3** | Drug labels / NCCN guidelines — **not** supported by `NCTLoader` |
| No source | **17** | Mostly brand-name combinations already resolved by hand (`CLIGAVYX/ribociclib`, `DONE ALREADY`) |

### 5.2 `drug_classes` — the target schema and the ground truth (38 rows, 28 columns)

This is the form a curator fills in manually today. The columns that matter for this
pipeline:

- **`Superclass`** — the primary target of Step 3
- **`Therapeutic Category`** (single value) — e.g. `antineoplastic_drug`
- **`Member of Drug Class`** — e.g. `small_molecule_drug`, `antibody_drug`
- **`Functions As`** — e.g. `inhibitor`, `antibody`
- **`Class Disambiguation`** — e.g. `MTA-cooperative PRMT5 inhibitor`
- **`Documentation`** — free text in exactly the format Step 1 produces

That last point is important. The `Documentation` column already contains strings like:

```
Subclass of proprietary_small_molecule; CAS: ...
Subclass of antineoplastic_combination_chemotherapy_regimen; ...
```

Step 1's required output format (`Subclass of <parent>. <definition>. <citation>`) was
designed to match it.

**This makes `drug_classes` simultaneously the output schema and the evaluation set.** For
`PEP08` the human answer is on record — `superclass = proprietary_small_molecule`,
`therapeutic_category = antineoplastic_drug`,
`class_disambiguation = MTA-cooperative PRMT5 inhibitor` — and can be compared directly
against what the pipeline produces.

### 5.3 Remaining sheets — other curation tasks, out of scope

| Sheet | Purpose | State |
|---|---|---|
| `DTEOs` | Drug–target–effect objects: drug × target protein × function | 3 rows |
| `Existing Class Edits` | Ontology mutations, coded `N/S/SR/CR/M/D` (superclass change, slot attach, slot removal, rename, merge, documentation) | 13 rows |
| `Synonym Moves` | Relocating a synonym from one class to another | Headers only |
| `New Terms` | Author usage → proposed mapping workflow | Headers only |
| `New Classes` | Requests for brand-new ontology classes | Headers only |
| `Definitions` | Definitional slot values per compound class | Headers only |
| `DIs` | Definitional instances (disease / cell / mutation modelling) | Headers only |

---

## 6. Output Format

`TherapeuticCategory.run()` returns a dictionary:

| Key | Type | Content |
|---|---|---|
| `term` | `str` | The input term, echoed back |
| `definition` | `str` | `Subclass of <parent>. <one sentence>. (ClinicalTrials.gov; <NCT ID>)` |
| `superclass_labels` | `dict` | `specific_label`, `broad_label`, `category_label` |
| `candidates` | `list[str]` | Up to 20 ontology terms retrieved from Qdrant, in score order |
| `ontology_mapping` | `dict` | `selected_term`, `confidence`, `match_type`, `reason` |
| `error` | `str` | Present **only** on failure; earlier keys are still populated |

`ontology_mapping` field values:

- **`selected_term`** — an exact candidate name, or `NONE_OF_THE_ABOVE`
- **`confidence`** — `high` / `medium` / `low`. Measures confidence that this is the *best
  available* mapping — **not** how specific the term is
- **`match_type`** — `exact` / `broader_available` / `nearest_available` / `none`
- **`reason`** — one grounded sentence of justification

---

## 7. Configuration

All settings come from environment variables via `.env` (loaded by `python-dotenv`).
`AppConfig.from_env()` assembles the full configuration tree.

```env
# LLM (internal proxy, OpenAI-compatible)
LLM_MODEL=gpt-5-mini
LLM_BASE_URL=http://aiproxy-rwc.ingenuity.com/gpt-5-mini
LLM_API_KEY=none
LLM_APP_ID=<your-app-id>

# SapBERT (local server started by the app)
SAPBERT_MODEL=sapbert
SAPBERT_BASE_URL=http://localhost:7997/v1
SAPBERT_API_KEY=none
SAPBERT_PORT=7997

# Qdrant vector store
QDRANT_URL=<your-qdrant-url>
QDRANT_COLLECTION=<collection-name>
QDRANT_API_KEY=<your-qdrant-key>

# Paths and serving
MODELS_DIR=./models
SERVER_PORT=8000
```

**Prerequisites:**

- Python ≥ 3.13
- A Qdrant collection containing ontology terms, embedded with SapBERT, exposing a named
  vector `sapbert_terminology` and a `display_name` payload field
- SapBERT weights downloaded locally (see below)

---

## 8. Running the Pipeline

**One-time model download:**

```sh
uv run scripts/download_models.py
```

**Interactive REPL:**

```sh
uv run main.py
```

```
Drug term:  lenvatinib
NCT IDs (comma-separated):  NCT01234567, NCT07654321
```

**Smoke test over hardcoded cases:**

```sh
uv run scripts/test_therapeutic_pipeline.py
```

**HTTP service** — `src/api/routes.py` defines the `Server` class; serve
`Server(AppConfig.from_env()).server` with uvicorn, then:

```sh
curl -X POST http://localhost:8000/run_pipeline \
  -H 'Content-Type: application/json' \
  -d '{"term": "PEP08", "nct_ids": ["NCT06973863"]}'
```

Note that the first startup is slow in both paths: SapBERT weights are loaded into memory
before anything else runs.

---

## 9. Current Gap: Excel Is Not Yet Wired In

`SCRIBE_terms.xlsx` is the intended input, but it is **not connected to any code**. It
appears in no import, it is untracked in git, and `src/processing/excel/` exists as an empty
package waiting for it. Today the pipeline only accepts terms typed one at a time.

To close the loop, three pieces are needed:

1. **`src/processing/excel/` — a reader.** Parse the `Requested Drug-Trials` sheet into
   `(term, nct_ids)` pairs. Must split `NCT Record ID` on `|`, and skip rows whose
   `Justification` is not `NCT`.
2. **`src/processing/pipeline/` — a batch runner.** Iterate the parsed rows through
   `TherapeuticCategory.run()`, accumulating both successes and per-row errors.
3. **A writer.** Emit results in `drug_classes` column shape so a curator can review and
   merge them into the existing workbook.

Two open decisions:

- **The 3 KANG rows** point at internal drug-label and NCCN-guideline PDFs. Supporting them
  requires a loader alongside `NCTLoader` — the current one only speaks
  ClinicalTrials.gov API v2.
- **The 17 source-less rows** are already-resolved brand combinations. They should probably
  be filtered out rather than processed.

Once the reader exists, `drug_classes` also becomes usable as an evaluation set: 38 rows of
human-curated superclass assignments to measure the pipeline against.
