from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

import toml

def modify_codex_config(username: str, tools_config: Path) -> None:
    """Modify the Codex config to include the MCP server for the given username and tools configuration."""
    codex_path = Path(os.path.expanduser("~/.codex/config.toml"))
    codex_path.parent.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = toml.load(codex_path)

    # Completely replace the bioinformatics MCP block
    mcp_block: dict[str, Any] = {
        "command": "/home/dionizije/.local/share/mamba/envs/bioinformatics-mcp/bin/python",
        "args": [
            "/home/dionizije/bioinformatics-mcp/mcp_server.py",
            "--no-dashboard",
            "--user",
            username,
            "--tool-config",
            str(tools_config),
        ],
        "startup_timeout_sec": 30,
        "tool_timeout_sec": 259200,
    }

    mcp_servers = config.setdefault("mcp_servers", {})
    mcp_servers["bioinformatics"] = mcp_block

    codex_path.write_text(toml.dumps(config), encoding="utf-8")


def remove_codex_mcp_config() -> None:
    """Remove the bioinformatics MCP block from the Codex config, if present."""
    codex_path = Path(os.path.expanduser("~/.codex/config.toml"))
    if not codex_path.exists():
        return

    config: dict[str, Any] = toml.load(codex_path)

    mcp_servers = config.get("mcp_servers")
    if isinstance(mcp_servers, dict) and "bioinformatics" in mcp_servers:
        del mcp_servers["bioinformatics"]
        if not mcp_servers:
            config.pop("mcp_servers", None)

        codex_path.write_text(toml.dumps(config), encoding="utf-8")

def modify_claude_config(username: str, tools_config: Path) -> None:
    remove_claude_mcp_config()
    subprocess.run(
        [
            "claude",
            "mcp",
            "add",
            "--transport",
            "stdio",
            "bioinformatics-mcp",
            "--",
            "/home/dionizije/.local/share/mamba/envs/bioinformatics-mcp/bin/python" \
            "/home/dionizije/bioinformatics-mcp/mcp_server.py" \
            "--no-dashboard" \
            "--user",
            username,
            "--tool-config",
            tools_config,
        ]
    )

def remove_claude_mcp_config() -> None:
    subprocess.run(
        [
            "claude",
            "mcp",
            "remove"
            "bioinformatics-mcp",
        ]
    )