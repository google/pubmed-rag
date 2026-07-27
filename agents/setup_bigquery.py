#!/usr/bin/env python3
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

"""One-time BigQuery provisioning for the PubMed agents.

Idempotent: safe to re-run. Creates the dataset, the remote embedding model,
and the journal-impact lookup table, then smoke-tests vector search.

    python agents/setup_bigquery.py
"""

import argparse
import logging
import pathlib
import sys

import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agents.core.config import EMBEDDING_DIMENSIONS, ConfigError, get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("setup")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "scimagojr_2025.csv"

# Columns kept from the SCImago export, renamed to BigQuery-friendly names.
JOURNAL_COLUMNS = {
    "Title": "journal_title",
    "sjr_float": "sjr",
    "Issn": "issn",
    "SJR Best Quartile": "sjr_best_quartile",
    "H index": "h_index",
    "Publisher": "publisher",
    "Categories": "categories",
    "Country": "country",
    "Type": "type",
}


def ensure_dataset(client: bigquery.Client, settings) -> None:
    ref = f"{settings.project_id}.{settings.bq_dataset}"
    try:
        client.get_dataset(ref)
        log.info("  dataset exists          %s", ref)
    except Exception:
        dataset = bigquery.Dataset(ref)
        # Must match the public dataset's location (US).
        dataset.location = settings.bq_location
        client.create_dataset(dataset, exists_ok=True)
        log.info("  dataset created         %s (%s)", ref, settings.bq_location)


def ensure_embedding_model(client: bigquery.Client, settings) -> None:
    """Create the remote embedding model.

    CONNECTION DEFAULT uses the auto-provisioned connection, so no connection
    or service-account grant is needed.
    """
    sql = f"""
    CREATE MODEL IF NOT EXISTS `{settings.embedding_model}`
      REMOTE WITH CONNECTION DEFAULT
      OPTIONS(endpoint='{settings.embedding_endpoint}')
    """
    client.query(sql).result()
    log.info(
        "  embedding model ready   %s (%s)",
        settings.embedding_model,
        settings.embedding_endpoint,
    )


def ensure_journal_table(
    client: bigquery.Client, settings, csv_path: pathlib.Path, force: bool
) -> None:
    ref = settings.journal_table
    if not force:
        try:
            table = client.get_table(ref)
            log.info("  journal table exists    %s (%d rows)", ref, table.num_rows)
            return
        except Exception:
            pass

    if not csv_path.exists():
        log.warning(
            "  journal CSV missing     %s -- skipping. Scoring will award "
            "0 journal-impact points.",
            csv_path,
        )
        return

    # SCImago ships semicolon-delimited with comma decimal separators.
    frame = pd.read_csv(csv_path, sep=";", dtype=str)
    frame["sjr_float"] = (
        frame["SJR"].str.replace(",", ".", regex=False).astype(float, errors="ignore")
    )
    frame["sjr_float"] = pd.to_numeric(frame["sjr_float"], errors="coerce")
    frame["H index"] = pd.to_numeric(frame["H index"], errors="coerce")

    clean = frame[list(JOURNAL_COLUMNS)].rename(columns=JOURNAL_COLUMNS)
    clean = clean[clean["sjr"].notna()]

    job = client.load_table_from_dataframe(
        clean,
        ref,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
    log.info("  journal table loaded    %s (%d rows)", ref, len(clean))


def smoke_test(client: bigquery.Client, settings) -> bool:
    sql = f"""
    WITH search_results AS (
        SELECT base.pmid, base.title, distance
        FROM VECTOR_SEARCH(
            TABLE `{settings.pubmed_table}`, 'ml_generate_embedding_result',
            (SELECT ml_generate_embedding_result FROM ML.GENERATE_EMBEDDING(
                MODEL `{settings.embedding_model}`,
                (SELECT 'high-risk neuroblastoma treatment' AS content))),
            top_k => 5)
    )
    SELECT pmid, title FROM search_results ORDER BY distance LIMIT 3
    """
    rows = list(client.query(sql).result())
    if not rows:
        log.error("  smoke test FAILED       vector search returned no rows")
        return False
    log.info("  smoke test OK           %d articles, top: %s",
             len(rows), (rows[0]["title"] or "")[:60])
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=pathlib.Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--force-journals", action="store_true", help="Reload the journal table"
    )
    args = parser.parse_args()

    try:
        settings = get_settings()
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    log.info("Provisioning BigQuery for %s", settings.project_id)
    client = bigquery.Client(project=settings.project_id)

    ensure_dataset(client, settings)
    ensure_embedding_model(client, settings)
    ensure_journal_table(client, settings, args.csv, args.force_journals)

    log.info("Verifying (%d-dim embeddings)...", EMBEDDING_DIMENSIONS)
    if not smoke_test(client, settings):
        return 1

    log.info("\nReady. Try:  adk run agents/pipelines/full_review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
