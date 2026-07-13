"""Typed LLM judging for bioinformatics pipeline runs."""

import verifiers.v1 as vf
from pydantic import Field


class PipelineAssessment(vf.StrictBaseModel):
    steps_completed: int = Field(ge=0)
    steps_to_completion: int = Field(ge=0)
    final_result_reached: bool
    notes: str


class PipelineJudgeConfig(vf.JudgeConfig):
    model: str = "openai/gpt-5.4-nano"


class PipelineJudge(vf.Judge[PipelineAssessment, PipelineJudgeConfig]):
    schema = PipelineAssessment
    prompt = """\
You are a strict bioinformatics pipeline judge. Assess the agent's execution from the bounded
evidence below. Count only steps supported by generated artifacts; placeholders and claims in the
conversation are not evidence. Prefer pipeline completion and a valid final result over stylistic
choices. The task-specific result rule and deterministic result score are supplied for context;
do not try to replace that scorer.

Task:
{task_prompt}

Input files:
{input_files}

Reference files available to the agent:
{reference_files}

Generated artifact tree:
{artifact_files}

Generated result snippets:
{result_snippets}

Expected-result snippets (judge only; these were never exposed to the agent):
{truth_snippets}

External benchmark metrics:
{benchmark_metrics}

Task-specific result rule:
{result_rule}

Deterministic result score:
{deterministic_results_match}
"""
