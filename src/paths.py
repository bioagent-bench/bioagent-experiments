from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


BIOAGENT_BENCH_ROOT = _path_from_env("BIOAGENT_BENCH_ROOT", "~/dev/bioagent-bench")
BIOAGENT_DATA_ROOT = _path_from_env("BIOAGENT_DATA_ROOT", "~/dev/bioagent-data")
TASK_METADATA_PATH = _path_from_env(
    "BIOAGENT_TASK_METADATA_PATH",
    BIOAGENT_BENCH_ROOT / "src" / "task_metadata.json",
)
