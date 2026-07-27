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

"""Full review: librarian -> analyst -> reporter.

Hardwired rather than LLM-routed, so order is deterministic and each stage's
output is visible. Wiring only; behaviour lives in catalog/ and core/.
"""

import pathlib
import sys

# `adk run` imports this directory as a top-level module.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from google.adk.agents import SequentialAgent  # noqa: E402

from agents.catalog.clinical_librarian import librarian_agent  # noqa: E402
from agents.catalog.evidence_analyst import analyst_agent  # noqa: E402
from agents.catalog.reporter import reporter_agent  # noqa: E402

root_agent = SequentialAgent(
    name="pubmed_full_review",
    description=(
        "Searches PubMed Central, appraises the results against an evidence "
        "rubric, and synthesizes a structured literature review."
    ),
    sub_agents=[librarian_agent, analyst_agent, reporter_agent],
)
