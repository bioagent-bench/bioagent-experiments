from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

@dataclass
class DataSet:
    task_id: str
    name: str
    description: str
    task_prompt: str

    path: str | None = None

    @classmethod
    def load_all(
        cls,
        metadata_path: str | Path = Path('~/bioagent-bench/src/task_metadata.json').expanduser(),
        data_root: str | Path = Path('~/bioagent-data').expanduser(),
    ) -> List["DataSet"]:
        """
        Load datasets from task_metadata.json and map local paths by folder name.

        - Scans data_root for subfolders and indexes them by the last folder name
        - For each task in metadata, sets path if a matching folder exists
        """
        metadata_path = Path(metadata_path).expanduser()
        data_root = Path(data_root).expanduser()

        tasks: List[dict] = json.loads(metadata_path.read_text())

        path_map = {
            p.name: str(p)
            for p in data_root.iterdir()
            if p.is_dir()
        }

        datasets: List[DataSet] = []
        for t in tasks:
            datasets.append(
                cls(
                    task_id=t["task_id"],
                    name=t["name"],
                    description=t["description"],
                    task_prompt=t["task_prompt"],
                    path=path_map.get(t["task_id"]),
                )
            )

        return datasets