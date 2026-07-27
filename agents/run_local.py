#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a pipeline locally and print each stage as it completes.

    python agents/run_local.py --pipeline full_review --case "..."
    python agents/run_local.py --pipeline single_agent --case-file notes.txt

Requires PROJECT_ID. Run agents/setup_bigquery.py first.
"""

import argparse
import asyncio
import importlib
import pathlib
import sys
import time

from google.genai import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agents.core.config import ConfigError, get_settings  # noqa: E402

PIPELINES = ("full_review", "single_agent")

# A real synthesis runs to tens of thousands of characters.
_MIN_REPORT_CHARS = 2000

EXAMPLE_CASE = (
    "4-year-old male with high-risk MYCN-amplified neuroblastoma, "
    "relapsed after induction chemotherapy and autologous stem cell transplant. "
    "Seeking evidence on salvage immunotherapy options."
)


def load_pipeline(name: str):
    module = importlib.import_module(f"agents.pipelines.{name}")
    return module.root_agent


async def run(pipeline: str, case: str, quiet: bool) -> int:
    from google.adk.runners import InMemoryRunner

    agent = load_pipeline(pipeline)
    runner = InMemoryRunner(agent=agent, app_name="pubmed_rag")
    session = await runner.session_service.create_session(
        app_name="pubmed_rag", user_id="local"
    )

    message = types.Content(role="user", parts=[types.Part(text=case)])
    started = time.time()
    stages: list[tuple[str, str]] = []
    tool_calls: list[str] = []

    async for event in runner.run_async(
        user_id="local", session_id=session.id, new_message=message
    ):
        author = getattr(event, "author", "?")
        content = getattr(event, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            if getattr(part, "function_call", None):
                tool_calls.append(f"{author} -> {part.function_call.name}()")
                if not quiet:
                    print(f"  [tool] {author} -> {part.function_call.name}()", flush=True)
            text = getattr(part, "text", None)
            if text and text.strip():
                stages.append((author, text))
                if not quiet:
                    print(f"\n{'=' * 78}\n### {author}  (+{time.time() - started:.1f}s)\n{'=' * 78}")
                    print(text.strip(), flush=True)

    elapsed = time.time() - started
    final_state = (
        await runner.session_service.get_session(
            app_name="pubmed_rag", user_id="local", session_id=session.id
        )
    ).state

    print(f"\n{'=' * 78}")
    print(f"pipeline      : {pipeline}")
    print(f"elapsed       : {elapsed:.1f}s")
    print(f"tool calls    : {len(tool_calls)}  {tool_calls}")
    print(f"stages emitted: {[a for a, _ in stages]}")
    print(f"state keys    : {sorted(final_state.keys())}")
    articles = final_state.get("articles") or []
    scored = final_state.get("scored_articles") or []
    print(f"articles      : {len(articles)} found, {len(scored)} scored")
    if scored:
        print(f"top score     : {scored[0].get('score')}  pmid={scored[0].get('pmid')}")
    # write_report owns this key, so it is the full artifact.
    report = final_state.get("final_report") or (stages[-1][1] if stages else "")
    print(f"final_report  : {len(report)} chars")

    # Require the rendered table and a plausible length, not just non-empty.
    problems = []
    if not scored:
        problems.append("no articles were scored")
    if "Scored Search Results" not in report:
        problems.append("report is missing the scored-results table")
    if len(report) < _MIN_REPORT_CHARS:
        problems.append(f"report is only {len(report)} chars (<{_MIN_REPORT_CHARS})")

    if problems:
        print("RESULT        : FAILED -- " + "; ".join(problems))
        print("=" * 78)
        return 1
    print("RESULT        : OK")
    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=PIPELINES, default="full_review")
    parser.add_argument("--case", default=EXAMPLE_CASE)
    parser.add_argument("--case-file", type=pathlib.Path)
    parser.add_argument("--quiet", action="store_true", help="Summary only")
    args = parser.parse_args()

    case = args.case_file.read_text() if args.case_file else args.case

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"project={settings.project_id} model={settings.model_id} "
          f"location={settings.location} pipeline={args.pipeline}")
    return asyncio.run(run(args.pipeline, case, args.quiet))


if __name__ == "__main__":
    raise SystemExit(main())
