import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from .judge_agent import EvaluationResults, EvaluationResultsGiab


@dataclass
class RunConfig:
    run_hash: str
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
    run_path: Path
    duration: float = 0.0
    eval_results: EvaluationResults | EvaluationResultsGiab | None = None
    steps: int = 0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    run_logs_root: Optional[Path] = None
    data_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    tools: List[Any] = field(default_factory=list, repr=False)

    def save_run_metadata(self, file_path: Optional[Path] = None) -> None:
        """Save metadata about the experiment run.

        Args:
            file_path (Path | None): Path to the JSON file where metadata is stored.
                If ``None``, ``self.metadata_path`` must be set.

        Returns:
            None: This method writes metadata to ``file_path``.
        """

        target_path = file_path or self.metadata_path
        if target_path is None:
            raise ValueError("metadata_path must be provided when saving run metadata.")

        def _serialize_eval_results(results: EvaluationResults | EvaluationResultsGiab | None) -> Any:
            if results is None:
                return None
            if is_dataclass(results):
                return asdict(results)
            return results

        metadata = {
            "run_hash": self.run_hash,
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
            "run_logs_root": str(self.run_logs_root) if self.run_logs_root else None,
            "data_path": str(self.data_path) if self.data_path else None,
            "run_path": str(self.run_path),
            "metadata_path": str(target_path),
        }

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.metadata_path = target_path

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
        if timestamp_raw is None:
            raise ValueError("Run metadata is missing 'timestamp'.")

        tool_names = payload.get("tool_names") or payload.get("tools") or []

        run_config = cls(
            run_hash=payload["run_hash"],
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
            run_path=Path(payload["run_path"]),
            duration=payload.get("duration", 0.0),
            eval_results=payload.get("eval_results"),
            steps=payload.get("steps", 0),
            input_tokens=payload.get("input_tokens", 0.0),
            output_tokens=payload.get("output_tokens", 0.0),
            run_logs_root=Path(payload["run_logs_root"]) if payload.get("run_logs_root") else None,
            data_path=Path(payload["data_path"]) if payload.get("data_path") else None,
            metadata_path=file_path,
        )

        return run_config
