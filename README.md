# bioagent-experiments

## Environment Setup

The evaluation system uses isolated mamba environments for each run. The LLM is provided with
Python and R installations from the start.

## Requirements
### Miniforge installation for mamba
https://github.com/conda-forge/miniforge

### uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

## Setup
1. The project is run through uv package manager
```bash
uv sync
```

2. Define where you want to store the run logs
```bash
export RUN_LOGS=/home/user/run_logs/
```


3. Run telemetry (Optional)
```bash
python -m phoenix.server.main serve
```
