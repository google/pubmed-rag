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

"""Deploy a pipeline to Vertex AI Agent Engine.

    python agents/deploy.py --pipeline full_review
    python agents/deploy.py --pipeline full_review --dry-run
    python agents/deploy.py --list
    python agents/deploy.py --delete <resource-name>

Agent Engine does not accept LOCATION=global; use a regional endpoint via
AGENT_ENGINE_LOCATION (default us-central1). The Gemini calls made by the agent
still honour LOCATION.
"""

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agents.core.config import ConfigError, get_settings  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PIPELINES = ("full_review", "single_agent")

# Mirrors agents/requirements.txt; resolved inside the managed container.
RUNTIME_REQUIREMENTS = [
    "google-adk>=2.5.0",
    "google-genai>=2.14.0",
    "google-cloud-bigquery>=3.42.0",
    "google-cloud-bigquery-storage>=2.27.0",
    "db-dtypes>=1.7.0",
    "pandas>=2.2.3",
    "pydantic>=2.9",
    # Required server-side: the AdkApp wrapper is unpickled with vertexai.
    "google-cloud-aiplatform[agent_engines]>=1.162.0",
]


def _agent_engine_location() -> str:
    return os.environ.get("AGENT_ENGINE_LOCATION", "us-central1")


def _staging_bucket(settings) -> str:
    return os.environ.get(
        "STAGING_BUCKET", f"gs://{settings.project_id}-agent-engine-staging"
    )


def project_number(project_id: str) -> str:
    """Resolve a project number."""
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = google.auth.transport.requests.AuthorizedSession(credentials)
    response = session.get(
        f"https://cloudresourcemanager.googleapis.com/v3/projects/{project_id}"
    )
    response.raise_for_status()
    return response.json()["name"].split("/")[-1]


def ensure_bucket(uri: str, location: str, project_id: str) -> None:
    """Create the staging bucket and grant the Vertex AI service agent access.

    Agent Engine reads the pickled agent as the service agent, so the grant is
    required, not optional.
    """
    from google.cloud import storage
    from google.cloud.exceptions import Conflict, NotFound

    name = uri.removeprefix("gs://").split("/")[0]
    client = storage.Client(project=project_id)
    try:
        bucket = client.get_bucket(name)
        print(f"  staging bucket exists   {uri}")
    except NotFound:
        try:
            bucket = client.create_bucket(name, location=location)
            print(f"  staging bucket created  {uri} ({location})")
        except Conflict:
            bucket = client.get_bucket(name)
            print(f"  staging bucket exists   {uri}")

    agent = (
        f"serviceAccount:service-{project_number(project_id)}"
        "@gcp-sa-aiplatform.iam.gserviceaccount.com"
    )
    policy = bucket.get_iam_policy(requested_policy_version=3)
    role = "roles/storage.objectAdmin"
    for binding in policy.bindings:
        if binding.get("role") == role and agent in binding.get("members", set()):
            print(f"  service agent access    already granted ({role})")
            return
    policy.bindings.append({"role": role, "members": {agent}})
    bucket.set_iam_policy(policy)
    print(f"  service agent access    granted {role} to {agent.split(':')[1]}")


# The deployed agent runs as the Reasoning Engine service agent
# (gcp-sa-aiplatform-re), which is a different identity from the AI Platform
# service agent that reads the staging bucket. It has no access to your
# BigQuery data or to Gemini until granted.
SERVICE_AGENT_ROLES = (
    "roles/bigquery.jobUser",       # run query jobs
    "roles/bigquery.dataViewer",    # read the journal table
    "roles/bigquery.connectionUser",  # use the connection behind the embedding model
    "roles/aiplatform.user",        # call Gemini
)


def reasoning_engine_agent(project_id: str) -> str:
    return (
        f"serviceAccount:service-{project_number(project_id)}"
        "@gcp-sa-aiplatform-re.iam.gserviceaccount.com"
    )


def ensure_service_agent_roles(project_id: str) -> None:
    """Grant the Reasoning Engine service agent what the deployed agent needs."""
    import google.auth
    import google.auth.transport.requests

    agent = reasoning_engine_agent(project_id)
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = google.auth.transport.requests.AuthorizedSession(credentials)
    base = f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}"

    policy = session.post(f"{base}:getIamPolicy", json={}).json()
    bindings = policy.setdefault("bindings", [])
    changed = []
    for role in SERVICE_AGENT_ROLES:
        binding = next((b for b in bindings if b.get("role") == role), None)
        if binding is None:
            bindings.append({"role": role, "members": [agent]})
            changed.append(role)
        elif agent not in binding.get("members", []):
            binding.setdefault("members", []).append(agent)
            changed.append(role)

    if not changed:
        print("  service agent roles     already granted")
        return

    response = session.post(f"{base}:setIamPolicy", json={"policy": policy})
    response.raise_for_status()
    print(f"  service agent roles     granted {', '.join(changed)}")
    # IAM is eventually consistent; a deploy that starts immediately can still
    # see the old policy.
    print("  waiting 60s for IAM propagation...")
    time.sleep(60)


def load_pipeline(name: str):
    import importlib

    return importlib.import_module(f"agents.pipelines.{name}").root_agent


def deploy(pipeline: str, dry_run: bool) -> int:
    import vertexai
    from vertexai import agent_engines

    settings = get_settings()
    location = _agent_engine_location()
    bucket = _staging_bucket(settings)

    print(f"project        : {settings.project_id}")
    print(f"agent engine   : {location}")
    print(f"model location : {settings.location}")
    print(f"pipeline       : {pipeline}")
    print(f"staging bucket : {bucket}")

    root_agent = load_pipeline(pipeline)
    print(f"loaded agent   : {root_agent.name}")

    if dry_run:
        print("\ndry run: agent imports and builds cleanly; nothing deployed.")
        return 0

    ensure_bucket(bucket, location, settings.project_id)
    ensure_service_agent_roles(settings.project_id)
    vertexai.init(
        project=settings.project_id, location=location, staging_bucket=bucket
    )

    app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)

    # extra_packages is resolved relative to the working directory, so the
    # package must be given as "agents" from the repo root.
    os.chdir(REPO_ROOT)

    print("\ndeploying (this typically takes several minutes)...")
    remote = agent_engines.create(
        agent_engine=app,
        display_name=f"pubmed-{pipeline.replace('_', '-')}",
        description=root_agent.description,
        requirements=RUNTIME_REQUIREMENTS,
        # Ship the whole package so imports resolve as they do locally.
        extra_packages=["agents"],
        env_vars={
            "PROJECT_ID": settings.project_id,
            "LOCATION": settings.location,
            "MODEL_ID": settings.model_id,
            "SCORING_MODEL_ID": settings.scoring_model_id,
            "BQ_DATASET": settings.bq_dataset,
            "BQ_LOCATION": settings.bq_location,
            "PUBMED_TABLE": settings.pubmed_table,
            "EMBEDDING_ENDPOINT": settings.embedding_endpoint,
        },
    )
    print(f"\ndeployed: {remote.resource_name}")
    return 0


def list_engines() -> int:
    import vertexai
    from vertexai import agent_engines

    settings = get_settings()
    vertexai.init(project=settings.project_id, location=_agent_engine_location())
    found = list(agent_engines.list())
    if not found:
        print("no agent engines deployed")
    for engine in found:
        print(f"{engine.resource_name}  {getattr(engine, 'display_name', '')}")
    return 0


def test_engine(resource_name: str | None) -> int:
    """Send one case to a deployed engine and check it produces a report."""
    import vertexai
    from vertexai import agent_engines

    settings = get_settings()
    vertexai.init(project=settings.project_id, location=_agent_engine_location())

    if not resource_name:
        found = list(agent_engines.list())
        if not found:
            print("no agent engines deployed")
            return 1
        resource_name = found[0].resource_name

    engine = agent_engines.get(resource_name)
    print(f"querying {resource_name}")

    case = (
        "4-year-old male with high-risk MYCN-amplified neuroblastoma, relapsed "
        "after induction chemotherapy. Seeking salvage immunotherapy evidence."
    )
    authors, text_len = [], 0
    for event in engine.stream_query(user_id="smoke-test", message=case):
        author = event.get("author") if isinstance(event, dict) else None
        if author and author not in authors:
            authors.append(author)
        parts = (event.get("content") or {}).get("parts", []) if isinstance(event, dict) else []
        for part in parts:
            if part.get("text"):
                text_len += len(part["text"])

    print(f"stages   : {authors}")
    print(f"text     : {text_len} chars")
    ok = len(authors) >= 3 and text_len > 2000
    print("RESULT   :", "OK" if ok else "FAILED")
    return 0 if ok else 1


def delete_engine(resource_name: str) -> int:
    import vertexai
    from vertexai import agent_engines

    settings = get_settings()
    vertexai.init(project=settings.project_id, location=_agent_engine_location())
    agent_engines.get(resource_name).delete(force=True)
    print(f"deleted {resource_name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=PIPELINES, default="full_review")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--delete", metavar="RESOURCE_NAME")
    parser.add_argument(
        "--test", nargs="?", const="", metavar="RESOURCE_NAME",
        help="Query a deployed engine end to end (defaults to the first one)",
    )
    args = parser.parse_args()

    try:
        if args.list:
            return list_engines()
        if args.test is not None:
            return test_engine(args.test or None)
        if args.delete:
            return delete_engine(args.delete)
        return deploy(args.pipeline, args.dry_run)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
