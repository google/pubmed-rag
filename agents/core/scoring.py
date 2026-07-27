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

"""Article appraisal: LLM metadata extraction plus a weighted rubric.

This is the single copy of the scoring engine. Both pipelines import it.
"""

import logging
import math
from datetime import datetime
from typing import Any

from . import llm
from .journals import lookup_sjr

logger = logging.getLogger(__name__)

DEFAULT_CRITERIA: list[dict[str, Any]] = [
    {"name": "journal_impact", "description": "High-impact journal (automatic SJR lookup)", "type": "special", "weight": 25},
    {"name": "year_penalty", "description": "Penalty per year old", "type": "special", "weight": -5},
    {"name": "event_match", "description": "Points per matching event", "type": "special", "weight": 15},
    {"name": "novelty", "description": "Presents novel/innovative findings or approaches", "type": "boolean", "weight": 10},
    {"name": "disease_match", "description": "Discusses the specific disease from the case", "type": "boolean", "weight": 70},
    {"name": "pediatric_focus", "description": "Focuses on pediatric patients", "type": "boolean", "weight": 50},
    {"name": "treatment_shown", "description": "Shows treatment efficacy or outcomes", "type": "boolean", "weight": 80},
    {"name": "drugs_tested", "description": "Tests or discusses specific drugs/therapies", "type": "boolean", "weight": 5},
    {"name": "clinical_trial", "description": "Is a clinical trial", "type": "boolean", "weight": 50},
    {"name": "review_article", "description": "Is a review article", "type": "boolean", "weight": -5},
    {"name": "case_report", "description": "Is a case report", "type": "boolean", "weight": 5},
    {"name": "case_series", "description": "Is a case series or series of case reports", "type": "boolean", "weight": 10},
    {"name": "cell_studies", "description": "Includes cell/in-vitro studies", "type": "boolean", "weight": 5},
    {"name": "animal_studies", "description": "Includes animal/mouse model studies", "type": "boolean", "weight": 10},
    {"name": "clinical_study", "description": "Is a clinical study (observational or interventional)", "type": "boolean", "weight": 15},
    {"name": "clinical_study_on_children", "description": "Is a clinical study specifically on children", "type": "boolean", "weight": 20},
]


def normalize_journal_score(sjr: float | None, max_points: float) -> float:
    """Log-scale SJR so a few dominant journals cannot swamp the rubric."""
    if not sjr or sjr <= 0:
        return 0.0
    return min(math.log(sjr + 1) * (max_points / 12), max_points)


def calculate_article_score(
    metadata: dict[str, Any], events_list: list[str]
) -> tuple[float, dict[str, Any]]:
    """Apply the weighted rubric. Returns (score, per-criterion breakdown)."""
    score = 0.0
    breakdown: dict[str, Any] = {}

    for criterion in DEFAULT_CRITERIA:
        name = criterion["name"]
        weight = criterion["weight"]
        ctype = criterion.get("type", "boolean")

        if name == "journal_impact":
            sjr = lookup_sjr(metadata.get("journal_title", "") or "")
            if sjr > 0:
                points = normalize_journal_score(sjr, weight)
                score += points
                breakdown["journal_impact"] = round(points, 2)

        elif name == "year_penalty":
            try:
                year = int(metadata.get("year", 0) or 0)
            except (ValueError, TypeError):
                year = 0
            if year > 0:
                points = weight * (datetime.now().year - year)
                score += points
                breakdown["year_penalty"] = points

        elif name == "event_match":
            events = metadata.get("actionable_events", []) or []
            if isinstance(events, str):
                events = [e.strip() for e in events.split(",") if e.strip()]
            matched = 0
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        if event.get("matches_query", False):
                            matched += 1
                    elif isinstance(event, str) and any(
                        q.lower() in event.lower() for q in events_list
                    ):
                        matched += 1
            if matched:
                points = matched * weight
                score += points
                breakdown["event_match"] = points

        elif ctype == "boolean":
            value = metadata.get(name, False)
            if value is True or str(value).lower() == "true":
                score += weight
                breakdown[name] = weight

        elif ctype == "numeric":
            value = metadata.get(name, 0)
            if value:
                points = weight * (float(value) / 100)
                score += points
                breakdown[name] = round(points, 2)

    return round(score, 2), breakdown


def _extraction_prompt(article: dict[str, Any], disease: str, events: list[str]) -> str:
    fields = [
        f"1. disease_match: Does the article discuss {disease}? (true/false)",
        "2. title: Article title",
        "3. journal_title: Extract the journal name from the article",
        "4. year: Publication year (integer)",
        f"5. actionable_events: List which of these events are mentioned: {events}",
        "6. paper_type: Type of study (Clinical Trial, Review, Case Report, etc.)",
        "7. key_findings: Brief summary of main findings (1-2 sentences)",
    ]
    index = 8
    for criterion in DEFAULT_CRITERIA:
        if criterion["type"] == "boolean":
            fields.append(f"{index}. {criterion['name']}: {criterion['description']} (true/false)")
            index += 1

    joined = "\n    ".join(fields)
    return f"""Analyze this medical research article for relevance to:
Disease: {disease}
Actionable Events: {', '.join(events)}

For this article, extract:
{joined}

Return as a single JSON object.

Article:
PMID: {article.get('pmid', 'N/A')}
Content: {(article.get('content') or '')[:3000]}
"""


def analyze_single_article(
    article: dict[str, Any], disease: str, events: list[str]
) -> dict[str, Any]:
    """Extract rubric metadata for one article. {} on failure."""
    try:
        return llm.generate_json(_extraction_prompt(article, disease, events))
    except Exception as exc:  # noqa: BLE001
        logger.warning("metadata extraction failed pmid=%s: %s", article.get("pmid"), exc)
        return {}


def score_articles(
    articles: list[dict[str, Any]], disease: str, concepts: list[str]
) -> list[dict[str, Any]]:
    """Extract metadata, apply the rubric, return articles sorted by score."""
    scored: list[dict[str, Any]] = []
    for article in articles:
        record = dict(article)
        bq_pmid = record.get("pmid")
        record.update(analyze_single_article(record, disease, concepts))
        if bq_pmid:
            # BigQuery's pmid wins; never let the model's guess through.
            record["pmid"] = bq_pmid
        score, breakdown = calculate_article_score(record, concepts)
        record["score"] = score
        record["score_breakdown"] = breakdown
        scored.append(record)

    scored.sort(key=lambda r: r.get("score", 0), reverse=True)
    logger.info(
        "scored %d articles, top=%s",
        len(scored),
        scored[0].get("score") if scored else None,
    )
    return scored
