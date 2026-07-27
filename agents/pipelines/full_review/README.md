# full_review

`clinical_librarian` → `evidence_analyst` → `reporter`, wired as a
`SequentialAgent`.

Use this when interim results matter. Each stage emits its own output, so the
caller sees the search criteria, then the ranked shortlist, then the report —
rather than waiting for one opaque answer.

```bash
python agents/run_local.py --pipeline full_review
adk web agents/pipelines/full_review
python agents/deploy.py --pipeline full_review
```

| | |
| :-- | :-- |
| **Final output** | `state["final_report"]` |
| **Stages visible** | 3 |
| **Typical run** | ~165 s, ~12 Gemini calls, 10 articles |

The order is hardwired rather than delegated to an LLM router, so the data flow
is deterministic. `agent.py` is only wiring; behaviour lives in
[`agents/catalog/`](../../catalog/) and [`agents/core/`](../../core/).
