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

"""SCImago journal impact (SJR) lookups."""

import functools
import logging
import re

from google.cloud import bigquery

from .config import get_settings

logger = logging.getLogger(__name__)


def normalize_title(title: str) -> str:
    """Fold a journal name to a comparable key.

    Lowercases, drops a leading article, spells out ampersands, collapses
    punctuation, so "The Lancet" and "Lancet" resolve to the same entry.
    """
    folded = title.lower().strip()
    folded = re.sub(r"^(the|a|an)\s+", "", folded)
    folded = folded.replace("&", " and ")
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return folded.strip()


@functools.lru_cache(maxsize=1)
def get_journal_dict() -> dict[str, float]:
    """Map journal title -> SJR score. {} if the table is missing."""
    settings = get_settings()
    client = bigquery.Client(project=settings.project_id)
    sql = (
        f"SELECT journal_title, sjr FROM `{settings.journal_table}` "
        "WHERE sjr IS NOT NULL"
    )
    try:
        rows = client.query(sql).result()
        mapping = {r["journal_title"]: r["sjr"] for r in rows}
        logger.info("loaded %d journal impact entries", len(mapping))
        return mapping
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "journal impact table unavailable (%s): %s. "
            "Scoring will award 0 journal-impact points. "
            "Run agents/setup_bigquery.py to create it.",
            settings.journal_table,
            exc,
        )
        return {}


@functools.lru_cache(maxsize=1)
def _normalized_index() -> dict[str, float]:
    """Folded title -> SJR; highest wins on collision."""
    index: dict[str, float] = {}
    for title, sjr in get_journal_dict().items():
        if not title or sjr is None:
            continue
        key = normalize_title(title)
        if key and sjr > index.get(key, 0.0):
            index[key] = sjr
    return index


def lookup_sjr(journal_title: str) -> float:
    """Exact match first, then normalized. 0.0 when unknown."""
    if not journal_title:
        return 0.0
    exact = get_journal_dict().get(journal_title)
    if exact:
        return exact
    return _normalized_index().get(normalize_title(journal_title), 0.0) or 0.0
