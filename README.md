# bioagent-experiments
Experiments for bioagent benchmark

```bash
docker build -t bioagent .
```
```bash
docker run -it -v $(pwd):/app bioagent
```
```bash
micromamba create -f environment.yaml
``` 
```bash
micromamba activate env-name
```

```bash
pip install 'smolagents[docker, openai]'
```.