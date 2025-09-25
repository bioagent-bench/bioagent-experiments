import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

from judge_agent import EvaluationResults, EvaluationResultsGiab


@dataclass
class RunConfig:
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
    timestamp_start: datetime = field(default_factory=datetime.now)
    timestamp_end: datetime | None = None
    eval_results: EvaluationResults | EvaluationResultsGiab | None = None
    input_tokens: float | None = None
    output_tokens: float | None = None

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
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "max_steps": self.max_steps,
            "planning_interval": self.planning_interval,
            "tools": [getattr(tool, "name", repr(tool)) for tool in self.tools],
            "system_prompt": self.system_prompt,
            "model": self.model,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "eval_results": _serialize_eval_results(self.eval_results),
        }

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
