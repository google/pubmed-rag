# single_agent

One `LlmAgent` holding all three catalog tools, deciding its own tool order.

Use this when the caller wants a single tool surface rather than visible
interim stages — Gemini Enterprise, for example, or any integration expecting
one agent with one answer.

```bash
python agents/run_local.py --pipeline single_agent
adk web agents/pipelines/single_agent
python agents/deploy.py --pipeline single_agent
```

| | |
| :-- | :-- |
| **Final output** | `state["final_report"]` |
| **Stages visible** | 1 |
| **Typical run** | ~150 s, ~12 Gemini calls, 10 articles |

This replaces the original 837-line `ge-adk-agent` with about twenty lines of
wiring. Same capability, but the search, scoring and synthesis logic now lives
in [`agents/core/`](../../core/) where both pipelines share one copy.
