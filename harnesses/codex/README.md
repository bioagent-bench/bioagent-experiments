# BioAgent Codex harness

This package exposes Verifiers' built-in v1 `CodexHarness` under the local
`bioagent-codex` package ID. Verifiers installs the configured Codex release inside each
rollout runtime and routes its Responses API traffic through the interception server.

Keeping this adapter as a package gives the workspace a stable place for future BioAgent-specific
Codex behavior without copying the upstream harness implementation.
