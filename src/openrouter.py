from __future__ import annotations

import os

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set")

    base_url = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)
