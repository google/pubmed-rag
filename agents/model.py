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

"""Model construction for catalog agents.

Pins project and region on the model itself. Passing a bare model-id string
would let the runtime pick the region, and Agent Engine's region does not serve
every model.
"""

from google.adk.models import Gemini

from agents.core import get_settings


def gemini(model_id: str | None = None) -> Gemini:
    settings = get_settings()
    return Gemini(
        model=model_id or settings.model_id,
        client_kwargs={
            "vertexai": True,
            "project": settings.project_id,
            "location": settings.location,
        },
    )
