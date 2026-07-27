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

"""Environment-driven settings. No default project ID."""

import functools
import os
from dataclasses import dataclass

# Query embeddings must be 768-dim to match the corpus.
DEFAULT_PUBMED_TABLE = "bigquery-public-data.pmc_open_access_commercial.articles"
DEFAULT_EMBEDDING_ENDPOINT = "text-embedding-005"
EMBEDDING_DIMENSIONS = 768


class ConfigError(RuntimeError):
    """Raised when required configuration is absent."""


@dataclass(frozen=True)
class Settings:
    project_id: str
    location: str
    bq_dataset: str
    bq_location: str
    bq_connection: str
    pubmed_table: str
    embedding_endpoint: str
    model_id: str
    scoring_model_id: str

    @property
    def embedding_model(self) -> str:
        return f"{self.project_id}.{self.bq_dataset}.textembed"

    @property
    def journal_table(self) -> str:
        return f"{self.project_id}.{self.bq_dataset}.journal_impact"


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set.\n"
            f"  export {name}=your-gcp-project-id\n"
            f"Or copy agents/.env.example to agents/.env and fill it in."
        )
    return value


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Resolve settings once per process."""
    return Settings(
        project_id=_required("PROJECT_ID"),
        location=os.environ.get("LOCATION", "global"),
        bq_dataset=os.environ.get("BQ_DATASET", "pubmed_demo"),
        bq_location=os.environ.get("BQ_LOCATION", "US"),
        bq_connection=os.environ.get("BQ_CONNECTION", "us.default"),
        pubmed_table=os.environ.get("PUBMED_TABLE", DEFAULT_PUBMED_TABLE),
        embedding_endpoint=os.environ.get(
            "EMBEDDING_ENDPOINT", DEFAULT_EMBEDDING_ENDPOINT
        ),
        model_id=os.environ.get("MODEL_ID", "gemini-3.6-flash"),
        scoring_model_id=os.environ.get("SCORING_MODEL_ID", "gemini-3.6-flash"),
    )
