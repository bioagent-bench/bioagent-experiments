# bioagent-experiments

## Environment Setup

The evaluation system uses isolated mamba environments for each run. The LLM is provided with
Python and R installations from the start.

## Requirements
### Miniforge installation for mamba
https://github.com/conda-forge/miniforge

### Codex
```bash
npm i -g @openai/codex
```

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
```bash
export AZURE_OPENAI_API_KEY=""
export ANTHROPIC_FOUNDRY_API_KEY=""
```

4. Install a mamba/micromamba installation to run bioinformatics-mcp from the agent
5. Install an isolated hap.py mamba environment to run germline variant calling bnechmarking
```bash
mamba create --name hap hap.py
```


## Canonical run
1. run_evals.py
```bash
python run_evals.py --suite open --reference-mode with
```

2. src/eval.py
3. evaluate_run_db.py

## Run tests for models
You can run this before running experiments to check if model connections work
```bash
python test_coding_framework_configs.py 
```


## Ablation
Type-1 = Corrupt
Type-2 = Decoy