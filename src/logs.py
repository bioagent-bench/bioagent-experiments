import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from .judge_agent import EvaluationResults, EvaluationResultsGiab


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(handler)


def _serialize_eval_results(results: EvaluationResults | EvaluationResultsGiab | None) -> Any:
    """We need to convert the result classes to JSON"""
    if results is None:
        return None
    if is_dataclass(results):
        return asdict(results)


@dataclass
class RunConfig:
    run_hash: str
    metadata_path: Path
    timestamp: datetime
    run_dir_path: Path
    data_path: Path
    otel_sink_path: Path
    task_id: str
    task_prompt: str
    num_tools: int
    tool_names: List[str]
    system_prompt: str
    system_prompt_name: str
    experiment_name: str
    model: str
    duration: float = 0.0
    eval_results: EvaluationResults | EvaluationResultsGiab | None = None
    steps: int = 0
    tools: List[Any] = field(default_factory=list, repr=False)
    error_type: str | None = None
    error_message: str | None = None
    otel_sink_host: str = "127.0.0.1:4317"

    def save_run_metadata(self) -> None:
        metadata = {
            "run_hash": self.run_hash,
            "metadata_path": str(self.metadata_path),
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "num_tools": self.num_tools,
            "tool_names": list(self.tool_names),
            "tools": list(self.tool_names),
            "system_prompt": self.system_prompt,
            "system_prompt_name": self.system_prompt_name,
            "experiment_name": self.experiment_name,
            "model": self.model,
            "duration": self.duration,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "eval_results": _serialize_eval_results(self.eval_results),
            "run_dir_path": str(self.run_dir_path),
            "data_path": str(self.data_path),
            "otel_sink_host": self.otel_sink_host,
            "otel_sink_path": str(self.otel_sink_path),
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load_run_metadata(cls, file_path: Path) -> "RunConfig":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        timestamp_raw = payload.get("timestamp")
        tool_names = payload.get("tool_names") or payload.get("tools") or []
        run_dir_path_raw = payload.get("run_dir_path")
        data_path_raw = payload.get("data_path")

        return cls(
            run_hash=payload["run_hash"],
            metadata_path=file_path,
            timestamp=datetime.fromisoformat(timestamp_raw),
            task_id=payload["task_id"],
            task_prompt=payload["task_prompt"],
            num_tools=payload.get("num_tools", len(tool_names)),
            tool_names=list(tool_names),
            system_prompt=payload["system_prompt"],
            system_prompt_name=payload["system_prompt_name"],
            experiment_name=payload["experiment_name"],
            model=payload["model"],
            duration=payload.get("duration", 0.0),
            eval_results=payload.get("eval_results"),
            error_type=payload.get("error_type"),
            error_message=payload.get("error_message"),
            run_dir_path=Path(run_dir_path_raw),
            data_path=Path(data_path_raw),
            otel_sink_host=payload.get("otel_sink_host"),
            otel_sink_path=Path(payload["otel_sink_path"])
        )
