# bioagent-experiments
Experiments for bioagent benchmark

```bash
docker build -t bioagent .
```
```bash
docker run -it -v $(pwd):/app bioagent
```

```bash
uv sync
```

Run telemetry 
```bash
python -m phoenix.server.main serve
```
