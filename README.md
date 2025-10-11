# bioagent-experiments
Experiments for bioagent benchmark

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
