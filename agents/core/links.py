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

"""Identifier rendering.

Some articles have a PMC record but no PubMed ID (pmid "0" or empty); those
fall back to a PMCID link rather than rendering a dead PubMed URL.
"""

PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{}/"
PMC_URL = "https://pmc.ncbi.nlm.nih.gov/articles/{}/"

_MISSING = "N/A"


def is_valid_pmid(pmid: object) -> bool:
    text = str(pmid or "").strip()
    return text.isdigit() and text != "0"


def is_valid_pmcid(pmcid: object) -> bool:
    text = str(pmcid or "").strip()
    if not text or text.lower() == "nan":
        return False
    return text.lower().startswith("pmc") or (text.isdigit() and text != "0")


def pmid_link(pmid: object) -> str:
    text = str(pmid or "").strip()
    return f"[{text}]({PUBMED_URL.format(text)})" if is_valid_pmid(text) else _MISSING


def pmcid_link(pmcid: object) -> str:
    text = str(pmcid or "").strip()
    return f"[{text}]({PMC_URL.format(text)})" if is_valid_pmcid(text) else _MISSING


def reference_link(article: dict) -> str:
    """Prefer a PMID link, fall back to PMCID, then N/A."""
    if is_valid_pmid(article.get("pmid")):
        return pmid_link(article["pmid"])
    if is_valid_pmcid(article.get("pmc_id")):
        return pmcid_link(article["pmc_id"])
    return _MISSING


def scored_table(articles: list[dict]) -> str:
    """Markdown table of title, score and a resolvable identifier link."""
    lines = ["| Title | Score | Reference |", "| :--- | :--- | :--- |"]
    for article in articles:
        title = article.get("title") or "Unknown"
        score = article.get("score", 0) or 0
        lines.append(f"| {title} | {score:.1f} | {reference_link(article)} |")
    return "\n".join(lines)
