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

"""Reporter: scored articles in, markdown literature review out.

Reads   state["scored_articles"], state["disease"], state["concepts"]
Writes  state["final_report"] (written by the tool, not by the model)
        output_key="reporter_note"
"""

from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext

from agents.core import build_report
from agents.model import gemini


def write_report(case_notes: str, tool_context: ToolContext) -> str:
    """Synthesize the scored articles into a structured literature review.

    Args:
        case_notes: The original case notes or research question.

    Returns:
        The full markdown report.
    """
    scored = tool_context.state.get("scored_articles")
    if not scored:
        return (
            "No scored articles in session state. The evidence_analyst must run "
            "appraise_articles first."
        )

    report = build_report(
        case_notes=case_notes,
        disease=tool_context.state.get("disease", ""),
        concepts=tool_context.state.get("concepts", []),
        scored_articles=scored,
    )
    # Authoritative copy: callers read the report from state, not from the
    # model's text output.
    tool_context.state["final_report"] = report
    return report


reporter_agent = LlmAgent(
    name="reporter",
    model=gemini(),
    description="Synthesizes scored articles into a structured literature review.",
    instruction="""You are a medical research reporter.

Call `write_report` once, passing the original case notes as `case_notes`.

The tool writes the finished report itself. After it returns, reply with a
single short line confirming the report is ready and how many articles it
covers. Do not restate the report.
""",
    tools=[write_report],
    # Only the model's acknowledgement; the report itself is in state.
    output_key="reporter_note",
)
