from pathlib import Path

import pytest
import verifiers.v1 as vf
from bioagent_opencode.harness import OpenCodeHarness, OpenCodeHarnessConfig
from bioagent_taskset.artifacts import runtime_files
from bioagent_taskset.scoring import score_results
from bioagent_taskset.taskset import (
    BioAgentConfig,
    BioAgentTaskConfig,
    BioAgentTaskset,
)
from verifiers.v1.harnesses.codex import CodexHarness, CodexHarnessConfig
from verifiers.v1.runtimes import SubprocessConfig, SubprocessRuntime


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data-root"
    task_root = data_root / "transcript-quant"
    (task_root / "data").mkdir(parents=True)
    (task_root / "results").mkdir()
    (task_root / "data" / "reads_1.fq.gz").write_bytes(b"reads")
    (task_root / "results" / "truth.tsv").write_text(
        "transcript_id\tcount\nENST1\t4\n", encoding="utf-8"
    )
    metadata = tmp_path / "tasks.json"
    metadata.write_text(
        """[
          {
            "task_id": "transcript-quant",
            "name": "Transcript quantification",
            "description": "Quantify transcripts.",
            "task_prompt": "Create truth.tsv.",
            "download_urls": {
              "data": [{"filename": "data.tar.gz", "url": "https://example.test/data"}],
              "reference_data": [],
              "results": []
            }
          }
        ]""",
        encoding="utf-8",
    )
    return metadata, data_root


def test_workspace_plugins_resolve_to_typed_configs() -> None:
    assert vf.taskset_config_type("bioagent-taskset") is BioAgentConfig
    assert vf.harness_config_type("bioagent-codex") is CodexHarnessConfig
    harness_config = CodexHarnessConfig(id="bioagent-codex")
    assert isinstance(vf.load_harness(harness_config), CodexHarness)
    assert vf.harness_config_type("bioagent-opencode") is OpenCodeHarnessConfig
    opencode_config = OpenCodeHarnessConfig(id="bioagent-opencode")
    assert isinstance(vf.load_harness(opencode_config), OpenCodeHarness)


def test_taskset_loads_typed_tasks(tmp_path: Path) -> None:
    metadata, data_root = write_fixture(tmp_path)
    taskset = BioAgentTaskset(
        BioAgentConfig(
            metadata_path=metadata,
            data_root=data_root,
            task_ids=["transcript-quant"],
            include_reference=False,
        )
    )

    [task] = taskset.load()

    assert task.data.task_id == "transcript-quant"
    assert task.data.input_files == ["reads_1.fq.gz"]
    assert task.data.reference_files == []
    assert "results/" in task.data.prompt_text


@pytest.mark.asyncio
async def test_task_lifecycle_stages_and_exports_results(tmp_path: Path) -> None:
    metadata, data_root = write_fixture(tmp_path)
    taskset = BioAgentTaskset(
        BioAgentConfig(
            metadata_path=metadata,
            data_root=data_root,
            include_reference=False,
            task=BioAgentTaskConfig(artifact_root=tmp_path / "artifacts"),
        )
    )
    [task] = taskset.load()
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
    )
    runtime = SubprocessRuntime(SubprocessConfig(), name=f"test-{trace.id}")

    await runtime.start()
    try:
        await task.setup(trace, runtime)
        assert (runtime.workdir / "data" / "reads_1.fq.gz").read_bytes() == b"reads"
        await runtime.write("results/answer.tsv", b"transcript_id\tcount\nENST1\t4\n")
        files = await runtime_files(runtime, "outputs", "results")
        assert await score_results(
            "transcript-quant",
            runtime,
            files,
            data_root / "transcript-quant" / "results",
            {},
        )
        await task.finalize(trace, runtime)
    finally:
        await runtime.stop()

    exported = Path(trace.info["artifacts"]["directory"]) / "results" / "answer.tsv"
    assert exported.read_text(encoding="utf-8") == "transcript_id\tcount\nENST1\t4\n"
