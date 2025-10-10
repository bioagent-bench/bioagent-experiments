import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from .judge_agent import EvaluationResults, EvaluationResultsGiab


def _serialize_eval_results(results: EvaluationResults | EvaluationResultsGiab | None) -> Any:
    """We need to convert the result classes to JSON"""
    if results is None:
        return None
    if is_dataclass(results):
        return asdict(results)
    return results


@dataclass
class RunConfig:
    run_hash: str
    metadata_path: Path
    timestamp: datetime
    task_id: str
    task_prompt: str
    max_steps: int
    planning_interval: int
    num_tools: int
    tool_names: List[str]
    system_prompt: str
    system_prompt_name: str
    experiment_name: str
    model: str
    duration: float = 0.0
    eval_results: EvaluationResults | EvaluationResultsGiab | None = None
    steps: int = 0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    run_dir_path: Optional[Path] = None
    data_path: Optional[Path] = None
    tools: List[Any] = field(default_factory=list, repr=False)

    def save_run_metadata(self) -> None:
        """Save metadata about the experiment run.

        Raises:
            ValueError: If ``self.metadata_path`` is not defined.

        Returns:
            None: This method writes metadata to ``self.metadata_path``.
        """
        metadata = {
            "run_hash": self.run_hash,
            "metadata_path": str(self.metadata_path),
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "max_steps": self.max_steps,
            "planning_interval": self.planning_interval,
            "num_tools": self.num_tools,
            "tool_names": list(self.tool_names),
            "tools": list(self.tool_names),
            "system_prompt": self.system_prompt,
            "system_prompt_name": self.system_prompt_name,
            "experiment_name": self.experiment_name,
            "model": self.model,
            "duration": self.duration,
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "eval_results": _serialize_eval_results(self.eval_results),
            "run_dir_path": str(self.run_dir_path),
            "data_path": str(self.data_path),
        }

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load_run_metadata(cls, file_path: Path) -> "RunConfig":
        """Load a ``RunConfig`` from a run metadata JSON file.

        Args:
            file_path (Path): Path to a run metadata JSON file.

        Returns:
            RunConfig: Deserialized run configuration populated from metadata.
        """

        payload = json.loads(file_path.read_text(encoding="utf-8"))
        timestamp_raw = payload.get("timestamp")

        tool_names = payload.get("tool_names") or payload.get("tools") or []

        run_dir_path_raw = payload.get("run_dir_path")
        data_path_raw = payload.get("data_path")

        run_config = cls(
            run_hash=payload["run_hash"],
            metadata_path=file_path,
            timestamp=datetime.fromisoformat(timestamp_raw),
            task_id=payload["task_id"],
            task_prompt=payload["task_prompt"],
            max_steps=payload["max_steps"],
            planning_interval=payload["planning_interval"],
            num_tools=payload.get("num_tools", len(tool_names)),
            tool_names=list(tool_names),
            system_prompt=payload["system_prompt"],
            system_prompt_name=payload["system_prompt_name"],
            experiment_name=payload["experiment_name"],
            model=payload["model"],
            duration=payload.get("duration", 0.0),
            eval_results=payload.get("eval_results"),
            steps=payload.get("steps", 0),
            input_tokens=payload.get("input_tokens", 0.0),
            output_tokens=payload.get("output_tokens", 0.0),
            run_dir_path=Path(run_dir_path_raw),
            data_path=Path(data_path_raw),
        )

        return run_config
