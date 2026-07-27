# Clinical Librarian

Turns clinical case notes into a PubMed Central literature search.

| | |
| :-- | :-- |
| **Reads** | nothing — this is an entry point |
| **Writes** | `state["articles"]`, `state["disease"]`, `state["concepts"]` |
| **output_key** | `search_summary` |
| **Tools** | `search_literature(disease, concepts)` |
| **Model** | `MODEL_ID` (default `gemini-3.6-flash`) |
| **Depends on** | `agents.core` only |

## What it does

Reads free-text case notes, extracts the primary disease and three to six
actionable concepts, then runs one BigQuery vector search against the
precomputed embeddings in the public PubMed Central table.

Results are deduplicated to one row per paper (`QUALIFY ROW_NUMBER()`), because
the corpus embeds each article in several chunks and the same paper would
otherwise appear repeatedly.

## Composing with it

It is the first stage of both shipped pipelines. Anything downstream only needs
to read `state["articles"]`:

```python
from agents.catalog.clinical_librarian import librarian_agent
from google.adk.agents import SequentialAgent

pipeline = SequentialAgent(
    name="search_only",
    sub_agents=[librarian_agent],
)
```

## Notes

Article bodies are truncated to 6000 characters before entering session state.
On Agent Engine that state is persisted, and unbounded full texts would bloat
every stored session.
