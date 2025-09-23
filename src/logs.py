from datetime import datetime
from typing import List, Any
from judge_agent import EvaluationResults, EvaluationResultsGiab


class RunConfig:
    max_steps: int
    planning_interval: int
    tools: List[Any]
    system_prompt: str
    model: str
    task_id: str
    task_prompt: str
    timestamp_start: datetime.now().isoformat()
    timestamp_end: datetime.now().isoformat() = None
    eval_results: EvaluationResults | EvaluationResultsGiab = None

def save_run_metadata(run_dir: Path, task_id: str, task_name: str, task_description: str, task_prompt: str):
    """Save metadata about the experiment run."""
    metadata = {
        "task_id": task_id,
        "task_name": task_name,
        "task_description": task_description,
        "task_prompt": task_prompt,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)