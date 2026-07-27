# Evidence Analyst

Appraises article quality against a fixed weighted rubric.

| | |
| :-- | :-- |
| **Reads** | `state["articles"]`, `state["disease"]`, `state["concepts"]` |
| **Writes** | `state["scored_articles"]` |
| **output_key** | `scoring_summary` |
| **Tools** | `appraise_articles()` |
| **Model** | `MODEL_ID` for reasoning, `SCORING_MODEL_ID` per article |
| **Depends on** | `agents.core` only |

## What it does

For each article: one Gemini call extracts structured metadata (study type,
journal, year, which concepts appear), then `agents.core.scoring` applies a
16-criterion weighted rubric in plain Python.

The split matters. The model only reports what the article says; the arithmetic
is deterministic code. The model never chooses a score, so runs are comparable
and the breakdown is auditable — every article carries a `score_breakdown`
showing which criterion contributed what.

## Rubric shape

| Kind | Examples |
| :-- | :-- |
| Special | `journal_impact` (log-normalized SJR), `year_penalty`, `event_match` |
| Boolean | `clinical_trial` +50, `treatment_shown` +80, `review_article` −5 |

Journal impact resolves through `agents.core.journals`, which folds titles
before lookup — SCImago lists "The Lancet" while an article prints "Lancet",
and matching raw strings silently drops the points.

## Composing with it

Requires `state["articles"]`. Pair it with any producer of that key, not just
the shipped librarian:

```python
SequentialAgent(name="score_only", sub_agents=[my_search_agent, analyst_agent])
```

If `state["articles"]` is absent the tool returns a message saying so rather
than raising, so a misordered pipeline degrades visibly instead of crashing.

## Cost

One Gemini call per article. A 10-article run is 10 calls and dominates the
pipeline's wall time (roughly 90 s of a 165 s full run).
