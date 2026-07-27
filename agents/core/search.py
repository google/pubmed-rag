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

"""BigQuery vector search over PubMed Central open-access articles."""

import logging
from typing import Any

from google.cloud import bigquery

from .config import get_settings
from .links import pmcid_link, pmid_link

logger = logging.getLogger(__name__)

# Articles are embedded in chunks; QUALIFY keeps one row per paper.
_SEARCH_SQL = """
WITH search_results AS (
    SELECT base.*, distance
    FROM VECTOR_SEARCH(
        TABLE `{pubmed_table}`, 'ml_generate_embedding_result',
        (SELECT ml_generate_embedding_result FROM ML.GENERATE_EMBEDDING(
            MODEL `{embedding_model}`, (SELECT @query AS content))),
        top_k => @top_k
    )
)
SELECT pmid, pmc_id, title, article_text AS content, distance
FROM search_results
WHERE (retracted IS NULL OR LOWER(retracted) != 'yes')
QUALIFY ROW_NUMBER() OVER(PARTITION BY COALESCE(pmid, title) ORDER BY distance) = 1
ORDER BY distance
LIMIT @limit
"""


def search_articles(
    disease: str, concepts: list[str], *, top_k: int = 20, limit: int = 10
) -> list[dict[str, Any]]:
    """Semantic search for articles matching a disease and concept list.

    top_k exceeds limit so dedup still yields `limit` distinct papers.
    """
    settings = get_settings()
    query_text = f"{disease} {' '.join(concepts)}".strip()
    sql = _SEARCH_SQL.format(
        pubmed_table=settings.pubmed_table, embedding_model=settings.embedding_model
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("query", "STRING", query_text),
            bigquery.ScalarQueryParameter("top_k", "INT64", top_k),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )
    logger.info(
        "vector search query=%r top_k=%d limit=%d table=%s",
        query_text, top_k, limit, settings.pubmed_table,
    )
    client = bigquery.Client(project=settings.project_id)
    rows = [dict(r) for r in client.query(sql, job_config=job_config).result()]
    logger.info("vector search returned %d articles", len(rows))
    return rows


def format_article_table(articles: list[dict[str, Any]]) -> str:
    """Markdown table of titles with resolvable PubMed / PMC links."""
    if not articles:
        return "No articles found."

    lines = ["| Title | PMID | PMCID |", "| :--- | :--- | :--- |"]
    for article in articles:
        title = article.get("title") or "Unknown"
        lines.append(
            f"| {title} | {pmid_link(article.get('pmid'))} "
            f"| {pmcid_link(article.get('pmc_id'))} |"
        )
    return "\n".join(lines)
