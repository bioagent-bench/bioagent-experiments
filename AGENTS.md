# Repository guide

This workspace uses `verifiers.v1`; do not add a second orchestration loop around it.

- Put benchmark data, lifecycle hooks, tools, and scoring in `tasksets/`.
- Put agent launch behavior in `harnesses/`.
- Put reusable LLM assessment logic in `judges/`.
- Add complete reproducible runs under `configs/eval/`.
- Keep tasksets harness-agnostic and use the `vf.Runtime` contract for rollout filesystem access.
- Export exactly one `vf.Taskset` or `vf.Harness` subclass through a plugin package's `__all__`.
- Pin harness and framework versions in evaluation configs.

Run `uv run pytest` and `uv run ruff check .` after changes.
