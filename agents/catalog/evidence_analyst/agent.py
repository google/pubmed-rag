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

"""Evidence Analyst: candidate articles in, scored articles out.

Reads   state["articles"], state["disease"], state["concepts"]
Writes  state["scored_articles"]
        output_key="scoring_summary"
"""

from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext

from agents.model import gemini
from agents.core.links import scored_table
from agents.core import score_articles as score_articles_core


def appraise_articles(tool_context: ToolContext) -> str:
    """Score every article found by the librarian against the rubric.

    Returns:
        A markdown table of the five highest-scoring articles.
    """
    articles = tool_context.state.get("articles")
    if not articles:
        return (
            "No articles in session state. The clinical_librarian must run "
            "search_literature first."
        )

    disease = tool_context.state.get("disease", "")
    concepts = tool_context.state.get("concepts", [])

    scored = score_articles_core(articles, disease, concepts)
    tool_context.state["scored_articles"] = scored

    table = scored_table(scored[:5])
    return f"Scored {len(scored)} articles. Top 5:\n\n{table}"


analyst_agent = LlmAgent(
    name="evidence_analyst",
    model=gemini(),
    description="Appraises article quality with a weighted evidence rubric.",
    instruction="""You are an evidence analyst.

Call `appraise_articles` once. It scores every article the librarian found,
using a fixed rubric (study design, journal impact, recency, concept match).

Report the returned table verbatim. The scores come from the rubric -- never
adjust, re-rank, or invent them, and never add articles that are not listed.
""",
    tools=[appraise_articles],
    output_key="scoring_summary",
)
