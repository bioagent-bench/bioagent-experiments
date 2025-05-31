from smolagents import CodeAgent, ToolCollection
from typing import List, Any, Optional
from common.models import (
    create_azure_model,
    create_llama_model,
    create_claude_model,
    create_gemini_model,
)
import logging

logger = logging.getLogger(__name__)

MODEL_SETUP_FUNC = {
    'azure': create_azure_model(),
    'llama': create_llama_model(),
    'claude': create_claude_model(),
    'gemini': create_gemini_model()
}


class SingleAgentWorkflow:
    """Base class for all bioagents to reduce code duplication."""

    def __init__(
        self,
        name: str = "bioagent",
        max_steps: int = 30,
        model_type: str = "llama",
        tools: Optional[List[Any]] = None,
        planning_interval: int = 1,
        add_base_tools: bool = True,
        additional_authorized_imports: Optional[List[str]] = None,
        executor_type: str = "local",
    ):
        """
        Initialize a bioagent with common configuration.

        Args:
            name: Name of the agent
            max_steps: Maximum number of steps for the agent to run
            model_type: Type of model to use (azure, llama, claude, gemini)
            tools: List of tools to provide to the agent
            planning_interval: How often to plan
            add_base_tools: Whether to add base tools
            additional_authorized_imports: Additional imports to authorize
            executor_type: Type of executor to use
        """
        # Set up model based on model_type
        if model_type not in MODEL_SETUP_FUNC:
            raise ValueError(f"Unknown model type: {model_type}")
        model_setup_func = MODEL_SETUP_FUNC[model_type]
        model = model_setup_func()

        # Set default values
        if additional_authorized_imports is None:
            additional_authorized_imports = ["*"]

        # TODO: Integrate Hydra for agent configs for experiments
        self.agent = CodeAgent(
            name=name,
            max_steps=max_steps,
            model=model,
            tools=tools,
            planning_interval=planning_interval,
            add_base_tools=add_base_tools,
            additional_authorized_imports=additional_authorized_imports,
            executor_type=executor_type,
        )

    def run(self, prompt: str) -> str:
        """
        Run the agent with the given prompt.

        Args:
            prompt: The prompt to run the agent with

        Returns:
            The result of running the agent
        """
        result = self.agent.run(prompt)
        return result

    @classmethod
    def run_with_prompt(cls, prompt: str, model_type: str = "llama", mcp_url: str = "http://0.0.0.0:8000/sse"):
        """
        Create an agent, run it with the given prompt, and print the result.

        This is a convenience method that handles the common pattern of:
        1. Creating a ToolCollection from MCP server
        2. Creating a BioAgent with the tools
        3. Running the agent with a prompt
        4. Printing the result

        Args:
            prompt: The prompt to run the agent with
            model_type: Type of model to use (azure, llama, claude, gemini)
            mcp_url: URL of the MCP server

        Returns:
            The result of running the agent
        """
        # Fetch tools from the MCP server
        with ToolCollection.from_mcp({"url": mcp_url}, trust_remote_code=True) as tool_collection:
            # Create the agent
            bioagent = cls(
                model_type=model_type,
                tools=tool_collection.tools,
            )

            result = bioagent.run(prompt)

            print(result)

            return result
