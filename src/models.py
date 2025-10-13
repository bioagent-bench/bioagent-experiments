from typing import Any

from openai import AzureOpenAI
from smolagents import AzureOpenAIServerModel, OpenAIServerModel


def load_keys(prefix):
    with open(f"/home/dionizije/bioagent-experiments/.keys/{prefix}_api.key", "r") as key_file:
        api_key = key_file.read().strip()

    with open(f"/home/dionizije/bioagent-experiments/.keys/{prefix}_endpoint.key", "r") as key_file:
        endpoint = key_file.read().strip()

    return api_key, endpoint


def create_azure_model(framework='smolagents'):
    api_key, endpoint = load_keys("azure")

    if framework == 'smolagents':
        return AzureOpenAIServerModel(
            model_id="gpt-5",
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2025-01-01-preview",
        )
    else:
        return AzureOpenAI(
            api_key=api_key,
            api_version="2025-01-01-preview",
            azure_endpoint=endpoint,
        )


def create_gemini_model():
    api_key, endpoint = load_keys("gemini")

    return OpenAIServerModel(
        model_id="gemini-2.0-flash",
        api_base=endpoint,
        api_key=api_key,
    )


def create_claude_model():
    api_key, endpoint = load_keys("claude")

    return OpenAIServerModel(
        model_id="claude-3-5-sonnet-20241022",
        api_base=endpoint,
        api_key=api_key,
    )


def create_llama_model():
    api_key, endpoint = load_keys("llama")

    return OpenAIServerModel(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        api_base=endpoint,
        api_key=api_key,
    )

model_loader_mapping = {
    'azure': create_azure_model,
    'llama': create_llama_model,
    'claude': create_claude_model,
    'gemini': create_gemini_model,
}


def load_model(model_name: str, **kwargs: Any) -> Any:
    """Load a model instance using the configured loader mapping.

    Args:
        model_name (str): Identifier for the desired model (e.g., ``"azure"``).
        **kwargs: Additional keyword arguments forwarded to the model loader.

    Returns:
        Any: Instantiated model client returned by the registered loader.
    """

    loader = model_loader_mapping.get(model_name)
    if loader is None:
        available = ", ".join(sorted(model_loader_mapping))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    return loader(**kwargs)