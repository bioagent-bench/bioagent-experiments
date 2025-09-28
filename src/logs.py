import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

from judge_agent import EvaluationResults, EvaluationResultsGiab


@dataclass
class RunConfig:
    timestamp: datetime
    task_id: str
    task_prompt: str
    max_steps: int
    planning_interval: int
    num_tools: int
    tools: List[Any]
    system_prompt: str
    system_prompt_name: str
    experiment_name: str
    model: str
    duration: float = 0.0
    eval_results: EvaluationResults | EvaluationResultsGiab | None = None
    steps: int = 0 
    input_tokens: float = 0.0
    output_tokens: float = 0.0

    def save_run_metadata(self, file_path: Path) -> None:
        """Save metadata about the experiment run.

        Args:
            file_path (Path): Path to the JSON file where metadata is stored.

        Returns:
            None: This method writes metadata to ``file_path``.
        """

        def _serialize_eval_results(results: EvaluationResults | EvaluationResultsGiab | None) -> Any:
            if results is None:
                return None
            if is_dataclass(results):
                return asdict(results)
            return results

        metadata = {
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "max_steps": self.max_steps,
            "planning_interval": self.planning_interval,
            "num_tools": self.num_tools,
            "tools": [getattr(tool, "name", repr(tool)) for tool in self.tools],
            "system_prompt": self.system_prompt,
            "system_prompt_name": self.system_prompt_name,
            "experiment_name": self.experiment_name,
            "model": self.model,
            "duration": self.duration,
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "eval_results": _serialize_eval_results(self.eval_results),
        }

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
