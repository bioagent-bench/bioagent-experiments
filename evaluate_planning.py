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
from src import models
from src.dataset import DataSet
from src.models import MODELS

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_LOGS = Path(os.getenv("RUN_LOGS"))
METADATA_PATH = Path("~/bioagent-bench/src/task_metadata.json").expanduser()
DATA_ROOT = Path("~/bioagent-data").expanduser()

PLAN_SCHEMA = """{
  "task_id": "string",
  "objective": "string",
  "steps": [
    {
      "step": "high-level step summary",
      "rationale": "why this step matters",
      "tool": "tool or system used"
    }
  ]
}"""

class PlanStepSchema(BaseModel):
    step: str
    rationale: str
    tool: str


class PlanSchema(BaseModel):
    task_id: str
    objective: str
    steps: list[PlanStepSchema] = Field(default_factory=list)


PLAN_EVAL_SCHEMA = """{
  "task_id": "string",
  "evaluations": [
    {
      "model_id": "string",
      "rating": 3,
      "rank": 2,
      "notes": "Two sentences: first about strengths, second about weaknesses."
    }
  ]
}"""


class PlanEvaluationItem(BaseModel):
    model_id: str
    rating: int
    rank: int
    notes: str


class PlanEvaluation(BaseModel):
    task_id: str
    evaluations: list[PlanEvaluationItem] = Field(default_factory=list)


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
    return f"""You are a model for planning bioinformatics pipelines. Use the task prompt and input data to produce a high-level overview plan.
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
    api_key_path = PROJECT_ROOT / ".keys" / "openrouter_api.key"
    api_key = api_key_path.read_text()
    api_url = PROJECT_ROOT / ".keys" / "openrouter_endpoint.key"
    api_url = api_url.read_text()
    return OpenAI(api_key=api_key, base_url=api_url)


def _plan_to_dict(plan: PlanSchema | dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan, dict):
        return plan
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    return plan.dict()


def _plan_eval_to_dict(plan_eval: PlanEvaluation | dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan_eval, dict):
        return plan_eval
    if hasattr(plan_eval, "model_dump"):
        return plan_eval.model_dump()
    return plan_eval.dict()


def run_planner(openai_client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    response = openai_client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=PlanSchema,
        # reasoning={"effort": "high"},
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


def write_plan(model: str, task_id: str, plan: dict) -> Path:
    plan_root = RUN_LOGS / "plans" / _model_slug(model)
    plan_root.mkdir(parents=True, exist_ok=True)
    plan_path = plan_root / f"{task_id}_{uuid.uuid4()}.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=True))
    return plan_path


def build_plan_eval_prompt(task_id: str, plans_by_model: dict[str, dict[str, Any]]) -> str:
    """Build the prompt for comparing plans across models for a single task.

    Args:
        task_id: Identifier of the task being evaluated.
        plans_by_model: Mapping from model identifier to its generated plan JSON.

    Returns:
        Prompt string to send to the LLM judge.
    """
    serialized_plans = {
        model_id: _plan_to_dict(plan) for model_id, plan in plans_by_model.items()
    }
    plans_json = json.dumps(serialized_plans, ensure_ascii=True, indent=2)
    return f"""You are evaluating alternative high-level bioinformatics pipeline plans for the same task.

For each model's plan, you must:
- Rate overall quality from 1 (very poor) to 5 (excellent).
- Assign a rank where 1 is the best plan, 2 is the second-best, etc.
- Write exactly two sentences of commentary:
  - First sentence: what is good or strong about the plan.
  - Second sentence: what is bad, risky, or missing from the plan.

When assigning ranks, do not leave ties: each model must have a unique rank.

Return ONLY a JSON object matching the PlanEvaluationSpec schema below. No markdown, no extra text.
All keys must be present. Use double quotes for all strings.

Task ID: {task_id}

CandidatePlans (model_id -> plan JSON):
{plans_json}

PlanEvaluationSpec JSON schema:
{PLAN_EVAL_SCHEMA}
"""


def run_plan_evaluator(openai_client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    """Call the LLM judge model to evaluate and rank plans.

    Args:
        openai_client: Configured OpenAI client.
        model: Model identifier for the judge.
        prompt: Prompt string describing the task and candidate plans.

    Returns:
        Parsed evaluation JSON as a dictionary.
    """
    response = openai_client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=PlanEvaluation,
    )
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return _plan_eval_to_dict(parsed)
    content = message.content or ""
    if content:
        return parse_plan_json(content)
    raise ValueError(f"Judge model {model} returned no content.")


def _filter_tasks(tasks: Iterable[DataSet], task_ids: Sequence[str]) -> list[DataSet]:
    if not task_ids:
        return list(tasks)
    wanted = set(task_ids)


def plan() -> None:
    openai_client = create_openai_client()

    datasets = DataSet.load_all(metadata_path=str(METADATA_PATH), data_root=str(DATA_ROOT))
    for task in list(datasets):
        if task.task_id not in ["single-cell", "transcript-quant", "viral-metagenomics"]:
            continue
        data_root = Path(task.path)
        data_dir = data_root / "data"
        reference_dir = data_root / "reference"
        input_data = glob_input_data(data_dir, reference_dir)

        prompt = build_plan_prompt(task.task_id, task.task_prompt, input_data)
        print(prompt)
        for model in MODELS:
            print(model)
            if model.startswith('openrouter'):
                model_name = model[len('openrouter/'):]
                print("Running planner")
                plan = run_planner(openai_client, model_name, prompt)
                write_plan(model, task.task_id, plan)


def evaluate_plans() -> None:
    """Evaluate and rank generated plans across models for each task.

    This reads the JSON plans written under ``RUN_LOGS / "plans"`` and,
    for each task, sends all candidate plans to a judge model that:

    - Rates each plan from 1–5.
    - Assigns a unique rank (1 = best).
    - Writes two sentences of notes per plan (strengths, then weaknesses).
    """
    openai_client = create_openai_client()
    judge_model = "openai/gpt-5.1"

    plans_root = RUN_LOGS / "plans"
    if not plans_root.exists():
        return

    # task_id -> model_slug -> plan_dict
    by_task: dict[str, dict[str, dict[str, Any]]] = {}

    for model_dir in plans_root.iterdir():
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        for plan_path in model_dir.glob("*.json"):
            try:
                plan_data = json.loads(plan_path.read_text())
            except json.JSONDecodeError:
                continue

            task_id = plan_data.get("task_id") or plan_path.stem.split("_", 1)[0]
            task_plans = by_task.setdefault(task_id, {})
            task_plans[model_id] = plan_data

    eval_root = RUN_LOGS / "plan_evals"
    eval_root.mkdir(parents=True, exist_ok=True)

    for task_id, plans_by_model in by_task.items():
        if not plans_by_model:
            continue

        prompt = build_plan_eval_prompt(task_id, plans_by_model)
        eval_result = run_plan_evaluator(openai_client, judge_model, prompt)

        eval_path = eval_root / f"{task_id}_{uuid.uuid4()}.json"
        eval_path.write_text(json.dumps(eval_result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    plan()
    evaluate_plans()
