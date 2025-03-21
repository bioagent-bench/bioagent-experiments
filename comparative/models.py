from smolagents import AzureOpenAIServerModel, OpenAIServerModel


def load_keys(prefix):
    with open(f"../keys/{prefix}_api.key", "r") as key_file:
        api_key = key_file.read().strip()

    with open(f"../keys/{prefix}_endpoint.key", "r") as key_file:
        endpoint = key_file.read().strip()

    return api_key, endpoint


def create_azure_model():
    api_key, endpoint = load_keys("azure")

    return AzureOpenAIServerModel(
        model_id="gpt-4o-mini",
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-12-01-preview",
    )


def create_llama_model():
    api_key, endpoint = load_keys("swiss")

    return OpenAIServerModel(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        api_base=endpoint,
        api_key=api_key,
    )
