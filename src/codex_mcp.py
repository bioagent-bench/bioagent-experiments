from __future__ import annotations

import os
from pathlib import Path
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