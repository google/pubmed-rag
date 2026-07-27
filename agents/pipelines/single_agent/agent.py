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

"""Single agent: one LlmAgent holding all three tools.

Same capability as full_review, exposed as one tool surface that decides its
own order. Use it when the caller wants one agent rather than visible stages.
"""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from google.adk.agents import LlmAgent  # noqa: E402

from agents.catalog.clinical_librarian import search_literature  # noqa: E402
from agents.catalog.evidence_analyst import appraise_articles  # noqa: E402
from agents.catalog.reporter import write_report  # noqa: E402
from agents.model import gemini  # noqa: E402

root_agent = LlmAgent(
    name="pubmed_single_agent",
    model=gemini(),
    description=(
        "Analyzes medical literature for a clinical case: searches PubMed "
        "Central, scores the evidence, and writes a literature review."
    ),
    instruction="""You analyze medical literature for clinical cases.

Given case notes, run these three steps in order:

  1. `search_literature(disease, concepts)` -- derive the disease and three to
     six concepts from the notes, then search.
  2. `appraise_articles()` -- score what you found.
  3. `write_report(case_notes)` -- synthesize the final review.

The tool writes the finished report itself. After step 3, reply with a single
short line confirming it is ready.

Never invent article titles, PMIDs, scores, or findings. Every claim must come
from a tool result. If a step returns no data, say so plainly and stop.
""",
    tools=[search_literature, appraise_articles, write_report],
    # No output_key: write_report already owns state["final_report"].
)
