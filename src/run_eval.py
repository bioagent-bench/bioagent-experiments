from docker_sandbox import DockerSandbox, ExecutionResult
from pathlib import Path
import json
from smolagents import CodeAgent
from models import create_azure_model
import os

tasks_paths = list(Path('~/bioagent-data').expanduser().glob('*'))
tasks = json.load(open(Path('~/bioagent-bench/src/task_metadata.json').expanduser()))

task_path_map = {
    p.name: p
    for p in Path("~/bioagent-data").expanduser().glob("*")
}

for task in tasks:
    task["path"] = task_path_map.get(task["task_id"])


def run_code_raise_errors(sandbox, code: str, verbose: bool = True) -> str:
    execution = sandbox.run_code(
        code,
        envs={'HF_TOKEN': os.getenv('HF_TOKEN')}
    )
    if execution.error:
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        logs = execution_logs
        logs += execution.error.traceback
        raise ValueError(logs)
    return "\n".join([str(log) for log in execution.logs.stdout])

for task in tasks:
    task_path = task['path']

    print(f"Processing task: {task['task_id']} at {task_path}")

    print(task['task_id'])

    agent = CodeAgent(
            max_steps=50,
            model=create_azure_model(),
            tools=[],
            planning_interval=1,
            additional_authorized_imports=["*"],
        )
    agent.run(task['task_prompt'])

