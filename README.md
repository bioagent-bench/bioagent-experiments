<img width="411" height="123" alt="image" src="https://github.com/user-attachments/assets/93c4b237-bb9f-4e38-b94a-218a093062ad" />

# BioAgent experiments

BioAgent evaluations now use the Verifiers v1 API introduced in `verifiers==0.2.0`.
The repository is a small workspace of composable packages:

```text
configs/eval/          complete evaluation recipes
harnesses/codex/       how Codex runs a task
harnesses/opencode/    how OpenCode runs a task
judges/pipeline/       typed pipeline assessment
tasksets/bioagent/     data, setup, artifacts, and scoring
```

The taskset is independent of Codex. The harness is independent of BioAgent data. Verifiers owns
the runtime lifecycle, API interception, traces, usage accounting, concurrency, timeouts, retries,
and cleanup.

## Setup

Requirements:

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- `OPENROUTER_API_KEY` for the supplied evaluation and judge endpoint
- BioAgent Bench metadata and downloaded task data
- `mamba` only when running tasks that install Conda packages; the `hap` environment is required
  for GIAB scoring

Install the workspace:

```bash
uv sync
```

The default paths are `~/dev/bioagent-bench/src/task_metadata.json` and
`~/dev/bioagent-data`. Edit [configs/eval/codex.toml](configs/eval/codex.toml), override its dotted
fields on the command line, or set `BIOAGENT_BENCH_ROOT`, `BIOAGENT_DATA_ROOT`, and
`BIOAGENT_TASK_METADATA_PATH` before loading a config that omits those path fields.

## Run

Validate package resolution and the full typed configuration without starting a rollout:

```bash
uv run eval @ configs/eval/codex.toml --dry-run
```

Run the configured evaluation:

```bash
uv run eval @ configs/eval/codex.toml
```

Run the pinned OpenCode recipe (configured for one cystic-fibrosis rollout with GLM 5.2):

```bash
uv run eval @ configs/eval/opencode.toml
```

For a smoke test, select one task and limit the run:

```bash
uv run eval @ configs/eval/codex.toml \
  --taskset.task-id transcript-quant \
  --num-tasks 1
```

Set `--taskset.include-reference false` to evaluate without reference resources. CLI values
override TOML values.

Verifiers writes the resolved config, trace records, and logs under `outputs/`. Before a runtime is
removed, the taskset copies final deliverables to `artifacts/<task-id>/<trace-id>/results/`. The
trace also contains the generated artifact tree, typed judge assessment, rewards, metrics, model
usage, judge usage, timings, and errors.

## Runtimes

The supplied config uses the subprocess runtime and stages already-downloaded inputs with symlinks.
This is fast for local experiments but gives the agent host access. To use Docker or Prime
sandboxes, set `taskset.staging = "download"` and select the corresponding
`harness.runtime.type`; each isolated runtime then downloads its task inputs from the benchmark
metadata.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Historical charts, result tables, and their analysis helpers remain in `results/`, `plotting/`, and
`utils/`. Install their optional dependencies with `uv sync --group analysis`.

Architecture and APIs follow the [Verifiers v1 launch
post](https://www.primeintellect.ai/blog/verifiers-v1) and the [v1
documentation](https://docs.primeintellect.ai/verifiers/v1/overview).
