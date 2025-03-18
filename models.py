from smolagents import AzureOpenAIServerModel, OpenAIServerModel


def create_azure_model():
    with open("api.key", "r") as key_file:
        api_key = key_file.read().strip()

    with open("endpoint.key", "r") as key_file:
        endpoint_key = key_file.read().strip()

    return AzureOpenAIServerModel(
        model_id="gpt-4o-mini",
        azure_endpoint=endpoint_key,
        api_key=api_key,
        api_version="2024-12-01-preview",
    )


swiss_model = OpenAIServerModel(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    api_base="https://fmapi.swissai.cscs.ch",
    api_key="sk-rc-9HZRysLKYoOZPBnDdhqVzw",
)
