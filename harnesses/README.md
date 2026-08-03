# Harnesses

Each subdirectory is an installable Verifiers v1 harness package. Harnesses own the agent
program and rollout strategy; they do not load benchmark data or score results.

Only `codex/` is enabled today. A future Claude Code integration should be added as a sibling
package rather than as a branch in the BioAgent taskset.
