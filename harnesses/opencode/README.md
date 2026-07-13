# BioAgent OpenCode harness

This package runs a pinned OpenCode release in non-interactive mode. It registers the Verifiers
interception server as an isolated OpenAI-compatible provider, so model traffic uses Chat
Completions while Verifiers retains tracing, limits, and usage accounting.

OpenCode state and configuration are isolated per rollout. Built-in tools run without interactive
approval, and task MCP servers are registered as remote MCP servers.
