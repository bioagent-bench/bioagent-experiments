# BioAgent taskset

The package converts BioAgent Bench metadata rows into typed Verifiers v1 tasks. Each rollout gets
an isolated workspace containing `data/`, optional `reference/`, `outputs/`, and `results/`.
Scoring happens while the runtime is still live, and final result files are exported before runtime
cleanup.
