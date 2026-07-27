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

"""Gemini access via the Google Gen AI SDK."""

import functools
import json
import logging
from typing import Any

from google import genai
from google.genai import types

from .config import get_settings

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(
        vertexai=True, project=settings.project_id, location=settings.location
    )


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_output_tokens: int = 8192,
) -> str:
    settings = get_settings()
    model_id = model or settings.model_id
    logger.info("generate_text model=%s prompt_chars=%d", model_id, len(prompt))
    response = get_client().models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_output_tokens
        ),
    )
    text = response.text or ""
    logger.info(
        "generate_text model=%s served=%s out_chars=%d",
        model_id,
        getattr(response, "model_version", "?"),
        len(text),
    )
    return text


def generate_json(
    prompt: str, *, model: str | None = None, temperature: float = 0.0
) -> dict[str, Any]:
    """Return parsed JSON, or {} if the model emits something unparseable."""
    settings = get_settings()
    model_id = model or settings.scoring_model_id
    response = get_client().models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature, response_mime_type="application/json"
        ),
    )
    raw = response.text or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "generate_json got unparseable payload model=%s raw=%r", model_id, raw[:500]
        )
        return {}
