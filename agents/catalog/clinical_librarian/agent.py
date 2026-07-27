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

"""Clinical Librarian: case notes in, candidate articles out.

Reads   nothing (entry point)
Writes  state["articles"], state["disease"], state["concepts"]
        output_key="search_summary"
"""

from google.adk.agents import LlmAgent
from google.adk.tools.tool_context import ToolContext

from agents.core import format_article_table, search_articles
from agents.model import gemini

# Bounded so persisted sessions stay small; the reporter reads 5000 chars.
_MAX_CONTENT_CHARS = 6000


def search_literature(
    disease: str, concepts: list[str], tool_context: ToolContext
) -> str:
    """Search PubMed Central for articles about a disease and related concepts.

    Args:
        disease: The primary condition, e.g. "high-risk neuroblastoma".
        concepts: Related search concepts, e.g. ["MYCN amplification"].

    Returns:
        A markdown table of the articles found.
    """
    articles = search_articles(disease, concepts)
    for article in articles:
        if article.get("content"):
            article["content"] = article["content"][:_MAX_CONTENT_CHARS]
        # numpy floats are not JSON-serializable.
        if article.get("distance") is not None:
            article["distance"] = float(article["distance"])

    tool_context.state["articles"] = articles
    tool_context.state["disease"] = disease
    tool_context.state["concepts"] = concepts

    if not articles:
        return "No articles found. Try broader concepts."
    return f"Found {len(articles)} articles.\n\n{format_article_table(articles)}"


librarian_agent = LlmAgent(
    name="clinical_librarian",
    model=gemini(),
    description="Turns clinical case notes into a PubMed literature search.",
    instruction="""You are a clinical librarian.

Given patient case notes, identify:
  1. The primary disease or condition.
  2. Three to six actionable concepts (mutations, biomarkers, treatments,
     patient population).

Then call `search_literature` once with those values.

Report the returned table verbatim. Do not invent article titles, PMIDs, or
findings, and do not evaluate article quality -- that is the next agent's job.
""",
    tools=[search_literature],
    output_key="search_summary",
)
