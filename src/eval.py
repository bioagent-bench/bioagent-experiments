
from pathlib import Path
from smolagents import CodeAgent
from models import create_azure_model
from dataset import DataSet
from system_prompts import system_prompt_v1
from judge_agent import parse_outputs

# Create output directories
def create_dirs(prefix: str):
    outputs_path = Path(prefix) / "outputs"
    results_path = Path(prefix) / "results"
    outputs_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

datasets = DataSet.load_all()

for task in datasets:
    print(f"Processing task: {task.task_id} at {task.path}")
    print(task.task_id)

    test_path = 'test_outputs/' + task.task_id
    # create_dirs(test_path)

    # agent = CodeAgent(
    #         max_steps=10,
    #         model=create_azure_model(),
    #         tools=[],
    #         additional_authorized_imports=["*"],
    #     )
    # agent.prompt_templates['system_prompt'] = system_prompt_v1
    # agent.run(task.task_prompt)

    output_tree = parse_outputs(test_path)
    print(output_tree)

