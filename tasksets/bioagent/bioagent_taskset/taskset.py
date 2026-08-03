"""BioAgent Bench as a composable Verifiers v1 taskset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from bioagent_judge import PipelineJudge, PipelineJudgeConfig

from bioagent_taskset.artifacts import (
    benchmark_metrics,
    export_results,
    list_local_files,
    local_table_snippets,
    runtime_files,
    runtime_table_snippets,
)
from bioagent_taskset.prompts import RESULT_RULES, SYSTEM_PROMPT
from bioagent_taskset.scoring import score_results

DEFAULT_BENCH_ROOT = Path(os.environ.get("BIOAGENT_BENCH_ROOT", "~/dev/bioagent-bench"))
DEFAULT_DATA_ROOT = Path(os.environ.get("BIOAGENT_DATA_ROOT", "~/dev/bioagent-data"))
DEFAULT_METADATA_PATH = Path(
    os.environ.get(
        "BIOAGENT_TASK_METADATA_PATH",
        str(DEFAULT_BENCH_ROOT / "src" / "task_metadata.json"),
    )
)

DOWNLOAD_SCRIPT = r"""\
import json
import tarfile
import urllib.request
from pathlib import Path

for category, items in json.loads(Path(".vf/downloads.json").read_text()).items():
    root = Path(category)
    root.mkdir(parents=True, exist_ok=True)
    for item in items:
        archive = root / item["filename"]
        urllib.request.urlretrieve(item["url"], archive)
        if tarfile.is_tarfile(archive):
            special = archive.name.split(".tar", 1)[0].removesuffix(".tgz")
            target = root / special if special.startswith(("kaiju_db_", "k2_standard_")) else root
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as bundle:
                bundle.extractall(target)
            archive.unlink()
"""


class Download(vf.StrictBaseModel):
    filename: str
    url: str


class BioAgentData(vf.TaskData):
    task_id: str
    source_dir: str
    staging: Literal["local", "download"]
    include_reference: bool
    data_downloads: list[Download]
    reference_downloads: list[Download]
    input_files: list[str]
    reference_files: list[str]
    result_rule: str


class BioAgentTaskConfig(vf.TaskConfig):
    judge: PipelineJudgeConfig = PipelineJudgeConfig()
    artifact_root: Path = Path("artifacts")


class BioAgentTask(vf.Task[BioAgentData, vf.State, BioAgentTaskConfig]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.run(["mkdir", "-p", "outputs", "results"], {})
        if self.data.staging == "local":
            source = Path(self.data.source_dir)
            categories = [
                "data",
                *(("reference",) if self.data.include_reference else ()),
            ]
            for category in categories:
                result = await runtime.run(
                    ["ln", "-s", str(source / category), category],
                    {},
                )
                if result.exit_code != 0:
                    raise RuntimeError(result.stderr)
            return

        manifest = {"data": [item.model_dump() for item in self.data.data_downloads]}
        if self.data.include_reference:
            manifest["reference"] = [item.model_dump() for item in self.data.reference_downloads]
        await runtime.write(".vf/download.py", DOWNLOAD_SCRIPT.encode())
        await runtime.write(".vf/downloads.json", json.dumps(manifest).encode())
        result = await runtime.run(["python", ".vf/download.py"], {})
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        files = await runtime_files(runtime, "outputs", "results")
        destination = self.config.artifact_root.expanduser() / self.data.task_id / trace.id
        await export_results(runtime, files, destination)
        trace.info["artifacts"] = {
            "directory": str(destination),
            "files": files,
        }

    @vf.reward(weight=1.0)
    async def results_match(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        artifact = trace.info["artifacts"]
        artifact_dir = Path(artifact["directory"])
        source_dir = Path(self.data.source_dir)
        benchmark = await benchmark_metrics(self.data.task_id, artifact_dir, source_dir)
        deterministic_match = await score_results(
            self.data.task_id,
            runtime,
            artifact["files"],
            source_dir / "results",
            benchmark,
        )
        result = await PipelineJudge(self.config.judge).evaluate(
            trace=trace,
            task_prompt=self.data.prompt_text,
            input_files=json.dumps(self.data.input_files, indent=2),
            reference_files=json.dumps(self.data.reference_files, indent=2),
            artifact_files=json.dumps(artifact["files"], indent=2),
            result_snippets=json.dumps(
                await runtime_table_snippets(runtime, artifact["files"]), indent=2
            ),
            truth_snippets=json.dumps(local_table_snippets(source_dir / "results"), indent=2),
            benchmark_metrics=json.dumps(benchmark, indent=2),
            result_rule=self.data.result_rule,
            deterministic_results_match=deterministic_match,
        )
        assessment = result.parsed
        trace.record_metrics(
            {
                "steps_completed": assessment.steps_completed,
                "steps_to_completion": assessment.steps_to_completion,
                "final_result_reached": assessment.final_result_reached,
                "results_match": deterministic_match,
            }
        )
        if "f1_score" in benchmark:
            trace.record_metric("f1_score", float(benchmark["f1_score"]))
        assessment_record = assessment.model_dump()
        assessment_record["results_match"] = deterministic_match
        trace.info["assessment"] = assessment_record
        return float(deterministic_match)


class BioAgentConfig(vf.TasksetConfig):
    metadata_path: Path = DEFAULT_METADATA_PATH
    data_root: Path = DEFAULT_DATA_ROOT
    task_id: str = ""
    task_ids: list[str] = []
    include_reference: bool = True
    staging: Literal["local", "download"] = "local"
    task: BioAgentTaskConfig = BioAgentTaskConfig()


class BioAgentTaskset(vf.Taskset[BioAgentTask, BioAgentConfig]):
    def load(self) -> list[BioAgentTask]:
        metadata_path = self.config.metadata_path.expanduser()
        data_root = self.config.data_root.expanduser()
        rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected = {*self.config.task_ids, *([self.config.task_id] if self.config.task_id else [])}

        tasks: list[BioAgentTask] = []
        for row in rows:
            task_id = row["task_id"]
            if selected and task_id not in selected:
                continue
            source_dir = data_root / task_id
            reference_downloads = row["download_urls"]["reference_data"]
            include_reference = self.config.include_reference and bool(reference_downloads)
            input_files = list_local_files(source_dir / "data") or [
                item["filename"] for item in row["download_urls"]["data"]
            ]
            reference_files = (
                list_local_files(source_dir / "reference")
                or [item["filename"] for item in reference_downloads]
                if include_reference
                else []
            )
            prompt = (
                f"{row['task_prompt'].strip()}\n\n"
                "Use the files in `data/`"
                + (" and `reference/`" if include_reference else "")
                + ". Write the requested final deliverable to `results/`."
            )
            tasks.append(
                BioAgentTask(
                    BioAgentData(
                        idx=len(tasks),
                        name=row["name"],
                        description=row["description"],
                        prompt=prompt,
                        system_prompt=SYSTEM_PROMPT,
                        task_id=task_id,
                        source_dir=str(source_dir),
                        staging=self.config.staging,
                        include_reference=include_reference,
                        data_downloads=[
                            Download.model_validate(item) for item in row["download_urls"]["data"]
                        ],
                        reference_downloads=[
                            Download.model_validate(item) for item in reference_downloads
                        ],
                        input_files=input_files,
                        reference_files=reference_files,
                        result_rule=RESULT_RULES[task_id],
                    ),
                    self.config.task,
                )
            )
        return tasks


__all__ = ["BioAgentTaskset"]
