#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI
from pydantic import BaseModel, Field
from src.dataset import DataSet
from src.models import MODELS

PROJECT_ROOT = Path(__file__).resolve().parent

PLAN_SCHEMA = """{
  "task_id": "string",
  "objective": "string",
  "stages": [
    {
      "id": "S1",
      "name": "stage name",
      "goal": "stage goal"
    }
  ],
  "steps": [
    {
      "id": "S1_step",
      "stage": "stage_id",
      "goal": "step goal",
      "inputs": ["input1", "input2"],
      "outputs": ["output1"],
      "key_decisions": [
        {"name": "decision_name", "options": ["opt1", "opt2"], "rationale": "why"}
      ],
      "checks": [
        {
          "name": "check_name",
          "pass_condition": "what must be true",
          "evidence": "artifact or metric"
        }
      ],
      "failure_modes": [
        {
          "symptom": "what goes wrong",
          "likely_causes": ["cause1", "cause2"],
          "mitigation": "how to fix"
        }
      ]
    }
  ],
  "final_artifact": {
    "type": "csv",
    "schema": [{"col": "column", "type": "string", "required": true}],
    "acceptance_criteria": ["criterion1", "criterion2"]
  },
  "assumptions": ["assumption1"],
  "constraints": ["constraint1"],
  "non_goals": ["non_goal1"]
}"""

class StageSchema(BaseModel):
    id: str
    name: str
    goal: str


class PlanSchema(BaseModel):
    task_id: str
    objective: str
    stages: list[StageSchema]
    steps: list[dict[str, Any]] = Field(default_factory=list)
    final_artifact: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)


def glob_input_data(*input_dirs: Path) -> list[Path]:
    files: set[Path] = set()
    for directory in input_dirs:
        root = Path(directory)
        if not root.exists():
            continue
        files.update(p.resolve() for p in root.rglob("*") if p.is_file())
    return sorted(files)


def build_plan_prompt(task_id: str, task_prompt: str, input_data: Sequence[Path]) -> str:
    input_list = json.dumps([str(path) for path in input_data], ensure_ascii=True)
    return f"""You are a model for planning bioinformatics pipelines. Use the task prompt and input data to produce a pipeline plan.
Return ONLY a JSON object matching the PipelinePlanSpec schema below. No markdown, no extra text.
All keys must be present. Use empty lists/objects when needed. Use double quotes for all strings.

Task ID: {task_id}
Task prompt: {task_prompt}
Input data files (absolute paths): {input_list}

PipelinePlanSpec JSON schema:
{PLAN_SCHEMA}
"""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = []
    for line in stripped.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_plan_json(raw: str) -> dict:
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])

def _read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    content = path.read_text().strip()
    return content or None


def create_openai_client() -> OpenAI:
    api_key = PROJECT_ROOT / ".keys" / "openrouter_api.key"
    base_url = PROJECT_ROOT / ".keys" / "openrouter_endpoint.key"
    return OpenAI(api_key=api_key, base_url=base_url)


def _plan_to_dict(plan: PlanSchema | dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan, dict):
        return plan
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    return plan.dict()


def run_planner(openai_client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    response = openai_client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=PlanSchema,
        reasoning={"effort": "high"},
    )
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return _plan_to_dict(parsed)
    content = message.content or ""
    if content:
        return parse_plan_json(content)
    raise ValueError(f"Model {model} returned no content.")

def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def write_plan(run_logs: Path, model: str, task_id: str, plan: dict) -> Path:
    plan_root = run_logs / "plans" / _model_slug(model) / task_id
    plan_root.mkdir(parents=True, exist_ok=True)
    plan_path = plan_root / f"{uuid.uuid4()}.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=True))
    return plan_path


def _filter_tasks(tasks: Iterable[DataSet], task_ids: Sequence[str]) -> list[DataSet]:
    if not task_ids:
        return list(tasks)
    wanted = set(task_ids)
    return [task for task in tasks if task.task_id in wanted]
    return parser.parse_args()


def main() -> None:
    run_logs = Path(args.run_logs)
    use_reference_data = args.reference_mode == "with"
    selected_models = args.models if args.models else MODELS
    openai_client = create_openai_client()

    datasets = DataSet.load_all(
        metadata_path=str(args.metadata_path),
        data_root=str(args.data_root),
    )
    tasks = _filter_tasks(datasets, args.task_ids or [])

    for task in tasks:
        if not task.path:
            continue
        data_root = Path(task.path)
        data_dir = data_root / "data"
        reference_dir = data_root / "reference"
        if use_reference_data:
            input_data = glob_input_data(data_dir, reference_dir)
        else:
            input_data = glob_input_data(data_dir)

        prompt = build_plan_prompt(task.task_id, task.task_prompt, input_data)
        for model in selected_models:
            plan = run_planner(openai_client, model, prompt)
            write_plan(run_logs, model, task.task_id, plan)


if __name__ == "__main__":
    main()
