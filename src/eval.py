
from pathlib import Path
from smolagents import CodeAgent
from models import create_azure_model
from dataset import DataSet
from system_prompts import system_prompt_v1

datasets = DataSet.load_all()

for task in datasets:
    print(f"Processing task: {task.task_id} at {task.path}")

    print(task.task_id)

    agent = CodeAgent(
            max_steps=2,
            model=create_azure_model(),
            tools=[],
            additional_authorized_imports=["*"],
        )
    agent.prompt_templates['system_prompt'] = system_prompt_v1
    # agent.run(task.task_prompt)
    agent.run("List all files in the current directory and print the results")
    
    print(error)

