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

"""Shared engine for every agent in the catalog.

Plain functions, no ADK imports, so they work from a notebook or a tool.
"""

from .config import ConfigError, Settings, get_settings
from .journals import get_journal_dict, lookup_sjr
from .links import pmcid_link, pmid_link, reference_link, scored_table
from .report import build_report
from .scoring import (
    DEFAULT_CRITERIA,
    analyze_single_article,
    calculate_article_score,
    normalize_journal_score,
    score_articles,
)
from .search import format_article_table, search_articles

__all__ = [
    "ConfigError",
    "Settings",
    "get_settings",
    "get_journal_dict",
    "lookup_sjr",
    "pmcid_link",
    "pmid_link",
    "reference_link",
    "scored_table",
    "build_report",
    "DEFAULT_CRITERIA",
    "analyze_single_article",
    "calculate_article_score",
    "normalize_journal_score",
    "score_articles",
    "format_article_table",
    "search_articles",
]
