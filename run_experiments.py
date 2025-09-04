from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
import os

from src.docker_sandbox import DockerSandbox
from src.dataset import DataSet

register()
SmolagentsInstrumentor().instrument()

def run_code_raise_errors(sandbox, code: str, verbose: bool = True) -> str:
    execution = sandbox.run_code(
        code,
    )
    if execution.error:
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        logs = execution_logs
        logs += execution.error.traceback
        raise ValueError(logs)
    return "\n".join([str(log) for log in execution.logs.stdout])


datasets = DataSet.load_all()

for dataset in datasets:
    agent_code = 

