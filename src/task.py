from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class DownloadItem:
    filename: str
    url: str


@dataclass
class DownloadURLs:
    data: List[DownloadItem]
    reference_data: List[DownloadItem]
    results: List[DownloadItem]


@dataclass
class Task:
    task_id: str
    name: str
    description: str
    task_prompt: str
    download_urls: DownloadURLs
    path: Optional[Path] = None

    @property
    def data_dir(self) -> Optional[Path]:
        return self.path


def _as_download_item(obj: Dict) -> DownloadItem:
    return DownloadItem(filename=obj["filename"], url=obj["url"])


def _as_download_urls(obj: Dict) -> DownloadURLs:
    return DownloadURLs(
        data=[_as_download_item(x) for x in obj.get("data", [])],
        reference_data=[_as_download_item(x) for x in obj.get("reference_data", [])],
        results=[_as_download_item(x) for x in obj.get("results", [])],
    )


def _as_task(obj: Dict, path_map: Optional[Dict[str, Path]] = None) -> Task:
    task_path = None
    if path_map is not None:
        task_path = path_map.get(obj.get("task_id"))
    return Task(
        task_id=obj["task_id"],
        name=obj["name"],
        description=obj["description"],
        task_prompt=obj["task_prompt"],
        download_urls=_as_download_urls(obj["download_urls"]),
        path=task_path,
    )


def load_tasks(
    metadata_path: Path = Path("~/bioagent-bench/src/task_metadata.json").expanduser(),
    data_root: Path = Path("~/bioagent-data").expanduser(),
) -> List[Task]:
    tasks_raw = json.loads(Path(metadata_path).read_text())
    path_map: Dict[str, Path] = {
        p.name: p for p in Path(data_root).glob("*") if p.is_dir()
    }
    return [_as_task(obj, path_map) for obj in tasks_raw]


def get_task_by_id(task_id: str, **kwargs) -> Optional[Task]:
    for task in load_tasks(**kwargs):
        if task.task_id == task_id:
            return task
    return None


