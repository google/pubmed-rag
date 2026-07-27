# PubMed Agents

A catalog of composable agents for evidence-based medicine literature review,
built on the [Agent Development Kit](https://google.github.io/adk-docs/) and
BigQuery vector search over PubMed Central.

Three agents. Two ready-made pipelines. One shared engine.

```
agents/
├── core/        the engine: search, scoring, journals, report. No ADK, no agents.
├── catalog/     one folder per role. Each has a README stating its contract.
└── pipelines/   thin wiring. full_review is 12 lines; single_agent is 20.
```

---

## Quickstart

Five commands from a clean checkout. Expect about ten minutes, most of it the
one-time BigQuery setup.

```bash
# 1. A Google Cloud project with billing enabled
export PROJECT_ID=your-gcp-project-id
gcloud auth application-default login
gcloud config set project "$PROJECT_ID"

# 2. Enable the two APIs this needs
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com

# 3. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r agents/requirements.txt

# 4. Provision BigQuery (idempotent, ~2 min; loads 31.7k journal impact scores)
python agents/setup_bigquery.py

# 5. Run
python agents/run_local.py --pipeline full_review
```

Step 5 prints each stage as it finishes and ends with a summary. A healthy run
looks like:

```
pipeline      : full_review
elapsed       : 166.0s
tool calls    : 3  ['clinical_librarian -> search_literature()', ...]
stages emitted: ['clinical_librarian', 'evidence_analyst', 'reporter']
articles      : 10 found, 10 scored
final_report  : 23341 chars
```

Exit code is 0 only if a report was actually produced, so this is safe in CI.

### Interactive

```bash
adk web agents/pipelines/full_review     # browser UI
adk run agents/pipelines/full_review     # terminal
```

---

## The catalog

| Agent | Reads | Writes | output_key |
| :-- | :-- | :-- | :-- |
| [`clinical_librarian`](catalog/clinical_librarian/) | — (entry point) | `articles`, `disease`, `concepts` | `search_summary` |
| [`evidence_analyst`](catalog/evidence_analyst/) | `articles`, `disease`, `concepts` | `scored_articles` | `scoring_summary` |
| [`reporter`](catalog/reporter/) | `scored_articles`, `disease`, `concepts` | `final_report` | `reporter_note` |

Each folder has a README with the full contract, cost, and composition notes.

### Pipelines

| Pipeline | Shape | Use when |
| :-- | :-- | :-- |
| [`full_review`](pipelines/full_review/) | `SequentialAgent` of all three | You want interim results visible — search criteria, then the ranked shortlist, then the report |
| [`single_agent`](pipelines/single_agent/) | One `LlmAgent` holding all three tools | The caller wants one tool surface, e.g. Gemini Enterprise |

Both produce the same final artifact and leave it at `state["final_report"]`.
The reporter's tool writes that key directly rather than relying on the model to
echo a 20k-character report, so `output_key` there only holds a short
acknowledgement.

---

## How composition works

Agents never call each other and never import each other. They communicate
through **session state**, and ADK's `SequentialAgent` runs them in order.

```
case notes
    │
    ├─ clinical_librarian ──writes──▶ state["articles"]
    │                                        │
    ├─ evidence_analyst ◀───reads────────────┘
    │        └──writes──▶ state["scored_articles"]
    │                              │
    └─ reporter ◀────reads─────────┘
             └──writes──▶ state["final_report"]
```

That is the entire contract. If an agent writes the key the next one reads,
they compose. This is why every catalog README leads with **Reads** and
**Writes** — those two lines are the interface.

Two consequences worth knowing:

- **Order is enforced by data, not by prompting.** `evidence_analyst` checks
  for `state["articles"]` and returns a plain message if it is missing, so a
  misordered pipeline degrades visibly rather than hallucinating a result.
- **Any stage is replaceable.** Write `state["scored_articles"]` with your own
  scorer and the shipped `reporter` works unchanged.

### Compose your own

Search and score, skip the write-up:

```python
from google.adk.agents import SequentialAgent
from agents.catalog.clinical_librarian import librarian_agent
from agents.catalog.evidence_analyst import analyst_agent

root_agent = SequentialAgent(
    name="triage_only",
    description="Find and rank literature without writing a full review.",
    sub_agents=[librarian_agent, analyst_agent],
)
```

Swap in your own scorer and keep the rest:

```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.tool_context import ToolContext
from agents.catalog.clinical_librarian import librarian_agent
from agents.catalog.reporter import reporter_agent

def my_scorer(tool_context: ToolContext) -> str:
    """Rank articles by recency instead of the evidence rubric."""
    articles = tool_context.state.get("articles", [])
    ranked = sorted(articles, key=lambda a: a.get("year", 0), reverse=True)
    for rank, article in enumerate(ranked):
        article["score"] = float(len(ranked) - rank)
    tool_context.state["scored_articles"] = ranked        # the contract
    return f"Ranked {len(ranked)} articles by recency."

recency_agent = LlmAgent(
    name="recency_scorer",
    model="gemini-3.6-flash",
    instruction="Call my_scorer once and report its output verbatim.",
    tools=[my_scorer],
    output_key="scoring_summary",
)

root_agent = SequentialAgent(
    name="recency_review",
    sub_agents=[librarian_agent, recency_agent, reporter_agent],
)
```

Save either as `agents/pipelines/<name>/agent.py` with an `__init__.py` that
does `from .agent import root_agent`, and `adk run agents/pipelines/<name>`
picks it up.

### Using the engine without agents

`agents/core` imports no ADK and knows nothing about agents. It is plain
functions, usable from a notebook or a script:

```python
from agents import core

articles = core.search_articles("high-risk neuroblastoma", ["MYCN amplification"])
scored   = core.score_articles(articles, "high-risk neuroblastoma", ["MYCN amplification"])
print(core.build_report("case notes here", "high-risk neuroblastoma",
                        ["MYCN amplification"], scored))
```

---

## Deploying

```bash
python agents/deploy.py --pipeline full_review --dry-run   # validate first
python agents/deploy.py --pipeline full_review
python agents/deploy.py --list
python agents/deploy.py --delete projects/.../reasoningEngines/123
```

The whole `agents/` package is shipped via `extra_packages`, so `core/`,
`catalog/` and `pipelines/` resolve on Agent Engine as they do locally.
Configuration is forwarded as environment variables, so the deployed agent
reads the same settings you ran with.

Agent Engine does not accept `LOCATION=global`. `deploy.py` uses
`AGENT_ENGINE_LOCATION` (default `us-central1`) for the engine while your
Gemini calls still honour `LOCATION`.

---

## Configuration

Only `PROJECT_ID` is required. Copy [`.env.example`](.env.example) to
`agents/.env` or export directly.

| Variable | Default | Notes |
| :-- | :-- | :-- |
| `PROJECT_ID` | **required** | No fallback. Missing values raise a named error |
| `LOCATION` | `global` | Vertex region for Gemini |
| `MODEL_ID` | `gemini-3.6-flash` | Agent reasoning |
| `SCORING_MODEL_ID` | `gemini-3.6-flash` | One call per article |
| `BQ_DATASET` | `pubmed_demo` | Created by `setup_bigquery.py` |
| `BQ_LOCATION` | `US` | Must be `US` — see below |
| `PUBMED_TABLE` | `bigquery-public-data.pmc_open_access_commercial.articles` | |
| `EMBEDDING_ENDPOINT` | `text-embedding-005` | Must be 768-dim — see below |
| `AGENT_ENGINE_LOCATION` | `us-central1` | Deploy only |
| `STAGING_BUCKET` | `gs://$PROJECT_ID-agent-engine-staging` | Deploy only; auto-created |

### Two settings you should not change casually

**`BQ_LOCATION` must be `US`.** The public PubMed dataset lives in the US
multi-region and BigQuery cannot vector-search across locations.

**`EMBEDDING_ENDPOINT` must produce 768 dimensions.** The corpus ships
precomputed 768-dimension embeddings in `ml_generate_embedding_result`. Your
query embedding has to match. `gemini-embedding-001` defaults to 3072 and would
require re-embedding 2.4 million rows of a read-only public table, which you
cannot do. `text-embedding-005` is current and not deprecated.

---

## Cost and timing

Measured on a 10-article run, `gemini-3.6-flash`, July 2026:

| Stage | Wall time | Gemini calls |
| :-- | :-- | :-- |
| Vector search | 5–10 s | 0 (BigQuery only) |
| Scoring | ~90 s | 1 per article |
| Report | ~60 s | 1 large (~60k input tokens) |
| **Total** | **~165 s** | **~12** |

Scoring dominates and scales linearly with article count. BigQuery cost is a
few cents per run; the public corpus is free to read.

---

## Troubleshooting

**`PROJECT_ID is not set`** — intentional. There is no default project, so a
misconfigured run fails immediately rather than pointing at someone else's
project.

**`does not have the permission to access or use the endpoint`** during
`setup_bigquery.py` — IAM propagation lag on the BigQuery connection's service
account. It resolves within a minute or two; re-run the script.

**Vector search returns irrelevant articles** — almost always an embedding
dimension mismatch. Confirm `EMBEDDING_ENDPOINT` is a 768-dimension model.

**All `journal_impact` scores are 0** — the `journal_impact` table is missing.
Scoring deliberately degrades instead of failing. Run
`python agents/setup_bigquery.py --force-journals`.

**`No articles in session state`** — a pipeline ran `evidence_analyst` before
`clinical_librarian`. Check `sub_agents` order.

---

## Layout

```
agents/
├── README.md               this file
├── requirements.txt
├── .env.example
├── setup_bigquery.py       one-time provisioning, idempotent
├── run_local.py            run a pipeline, print each stage
├── deploy.py               Vertex AI Agent Engine
├── core/
│   ├── config.py           env-driven settings, no hardcoded project
│   ├── llm.py              Gemini via google-genai
│   ├── search.py           BigQuery vector search
│   ├── scoring.py          metadata extraction + weighted rubric
│   ├── journals.py         SJR lookup with title folding
│   ├── links.py            PMID/PMCID rendering
│   └── report.py           11-section synthesis
├── catalog/
│   ├── clinical_librarian/
│   ├── evidence_analyst/
│   └── reporter/
└── pipelines/
    ├── full_review/
    └── single_agent/
```

---

## Credits

This catalog is built on earlier work in this repository.

- **[@gabbyburke](https://github.com/gabbyburke)** designed the multi-agent
  pipeline in [#2](https://github.com/google/pubmed-rag/pull/2) — the
  librarian, analyst and reporter split, and the tool implementations that
  became `agents/core/`. The three agents in `catalog/` are her design.
- **[@siduojiang](https://github.com/siduojiang)** (Stone Jiang) wrote the
  original `ge-adk-agent` in
  [#1](https://github.com/google/pubmed-rag/pull/1), including the weighted
  scoring rubric and journal-impact logic that `agents/core/scoring.py`
  carries forward largely unchanged.
