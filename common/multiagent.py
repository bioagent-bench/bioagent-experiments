from smolagents import CodeAgent, tool, ToolCollection
from typing import Dict, List, Any, Optional
from common.models import create_azure_model, create_llama_model, create_claude_model, create_gemini_model, model_loader_mapping
import subprocess
import os
import yaml
import logging
try:
    from phoenix.otel import register  # type: ignore
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor  # type: ignore
    register()
    SmolagentsInstrumentor().instrument()
except Exception:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@tool
def setup_environment(environment_spec: Dict[str, Any]) -> str:
    """
    Set up a conda environment based on the provided specification.

    Args:
        environment_spec: A dictionary containing environment name and packages to install

    Returns:
        Output from the environment setup process
    """
    env_name = environment_spec.get("name", "bioenv")
    packages = environment_spec.get("packages", [])
    channels = environment_spec.get("channels", ["conda-forge", "bioconda", "defaults"])

    env_config = {
        "name": env_name,
        "channels": channels,
        "dependencies": packages
    }

    with open("../deseq/environment.yaml", "w") as f:
        yaml.dump(env_config, f)

    try:
        result = subprocess.run(
            ["conda", "env", "create", "-f", "environment.yaml"],
            capture_output=True,
            text=True,
            check=True
        )
        return f"Environment {env_name} created successfully: {result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error creating environment: {e.stderr}"

@tool
def install_packages(env_name: str, packages: List[str]) -> str:
    """
    Install additional packages in an existing conda environment.

    Args:
        env_name: Name of the conda environment
        packages: List of packages to install

    Returns:
        Output from the package installation process
    """
    try:
        result = subprocess.run(
            ["conda", "install", "-n", env_name, "-y"] + packages,
            capture_output=True,
            text=True,
            check=True
        )
        return f"Packages installed successfully in {env_name}: {result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error installing packages: {e.stderr}"

@tool
def pip_install(env_name: str, packages: List[str]) -> str:
    """
    Install Python packages using pip in a conda environment.

    Args:
        env_name: Name of the conda environment
        packages: List of packages to install

    Returns:
        Output from the pip installation process
    """
    packages_str = " ".join(packages)
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install"] + packages_str,
            capture_output=True,
            text=True,
            check=True
        )
        return f"Pip packages installed successfully in {env_name}: {result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error installing pip packages: {e.stderr}"


# Define tools for the code agent
@tool
def execute_code(code: str, env_name: Optional[str] = None) -> str:
    """
    Execute the provided Python code.

    Args:
        code: Python code to execute
        env_name: Optional conda environment name to run the code in

    Returns:
        Output from the code execution
    """
    with open("temp_script.py", "w") as f:
        f.write(code)

    try:
        if env_name:
            result = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "temp_script.py"],
                capture_output=True,
                text=True,
                check=True
            )
        else:
            result = subprocess.run(
                ["python", "temp_script.py"],
                capture_output=True,
                text=True,
                check=True
            )
        return f"Code executed successfully: {result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error executing code: {e.stderr}"


class BioinformaticsMultiAgentWorkflow:
    def __init__(self, executor_type: str = "local"):
        self.model = create_azure_model()
        self.mcp_params = {"url": "http://0.0.0.0:8000/sse"}
        self.executor_type = executor_type

        # TODO: Hydra for configs
        with ToolCollection.from_mcp(self.mcp_params, trust_remote_code=True) as tool_collection:
            self.orchestrator = CodeAgent(
                name="orchestrator",
                max_steps=10,
                model=self.model,
                tools=list(tool_collection.tools),
                planning_interval=1,
                add_base_tools=True,
                executor_type=self.executor_type,
            )

        with ToolCollection.from_mcp(self.mcp_params, trust_remote_code=True) as tool_collection:
            self.environment_manager = CodeAgent(
                name="environment_manager",
                max_steps=10,
                model=self.model,
                tools=list(tool_collection.tools) + [setup_environment, install_packages, pip_install],
                planning_interval=1,
                add_base_tools=True,
                executor_type=self.executor_type,
            )

        with ToolCollection.from_mcp(self.mcp_params, trust_remote_code=True) as tool_collection:
            self.code_agent = CodeAgent(
                name="code_agent",
                max_steps=20,
                model=self.model,
                tools=list(tool_collection.tools) + [execute_code],
                planning_interval=1,
                add_base_tools=True,
                additional_authorized_imports=["*"],
                executor_type=self.executor_type,
            )

    def create_environment_setup_plan_prompt(self, execution_plan: str) -> str:
        return f"""You are an environment manager agent responsible for setting up bioinformatics environments.
        Based on the following execution plan, create and set up a conda environment with all necessary dependencies:

        {execution_plan}

        Use the tools available to you to create the environment.
        Return the name of the created environment and a list of installed packages.
        """

    def create_execution_plan_prompt(self, task_description: str) -> str:
        return f"""You are an orchestrator agent responsible for planning the execution of bioinformatics tasks.
        Based on the following task description, create a detailed execution plan:

        {task_description}

        Your execution plan should include:

        1. A list of steps to be performed
        2. Required tools and libraries for each step
        3. Expected inputs and outputs for each step

        Make sure that the plan you created is clear, detailed, concise, self-contained, easy to follow and easy to implement.
        """

    def create_code_execution_prompt(self, execution_plan: str, environment_setup) -> str:
        return f"""You are a code agent responsible for implementing and executing bioinformatics workflows.
        Based on the following execution plan, generate and execute code to solve the task:

        {execution_plan}

        The environment has been set up with the following configuration:
        
        {environment_setup}

        Generate Python code to implement each step of the plan and use the tools available to you to run it.
        Return the final results of the analysis.
        """


    def run(self, task_description: str):
        """
        Run the multiagent workflow.

        Args:
            task_description: Description of the bioinformatics task

        Returns:
            The final result of the workflow
        """
        logger.info("Starting multiagent workflow")


        # Ensure containers are cleaned up by using context managers
        with self.orchestrator as orchestrator, self.environment_manager as environment_manager, self.code_agent as code_agent:
            # Step 1. Creating an execution plan
            orchestrator_prompt = self.create_execution_plan_prompt(task_description)
            execution_plan = orchestrator.run(orchestrator_prompt)

            # Step 2: Orchestrator supplies environment setup plan to environment manager
            environment_setup_prompt = self.create_environment_setup_plan_prompt(execution_plan)
            environment_manager.run(environment_setup_prompt)
            environment_setup = environment_manager.run(environment_setup_prompt)

            # Step 3: Code generates and executes code based on the execution plan and environment setup
            code_agent_prompt = self.create_code_execution_prompt(execution_plan, environment_setup)
            code_execution_result = code_agent.run(code_agent_prompt)

        return code_execution_result