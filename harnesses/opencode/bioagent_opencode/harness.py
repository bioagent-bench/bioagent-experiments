"""Run OpenCode against the Verifiers interception server."""

from __future__ import annotations

import json
import logging
import shlex

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

PROVIDER = "intercept"
KEY_VAR = "VF_OPENCODE_KEY"
OPENCODE_DIR = "/tmp/vf-opencode"
OPENCODE_BIN = f"{OPENCODE_DIR}/bin/opencode"

INSTALL = r"""
set -e
mkdir -p {dir}/bin
if ! command -v curl >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates >/dev/null
fi
case "$(uname -m)" in aarch64|arm64) arch=arm64 ;; *) arch=x64 ;; esac
release="https://github.com/anomalyco/opencode/releases/download/v{version}"
curl -fsSL "${release}/opencode-linux-${arch}.tar.gz" \
    | tar -xzf - -C {dir}/bin
chmod +x {bin}
"""


class OpenCodeHarnessConfig(HarnessConfig):
    version: str = "1.15.11"
    """OpenCode release to install, pinned for reproducibility."""


class OpenCodeHarness(Harness[OpenCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info("opencode: ensuring OpenCode %s is installed", self.config.version)
        script = (
            INSTALL.replace("{version}", self.config.version)
            .replace("{dir}", OPENCODE_DIR)
            .replace("{bin}", OPENCODE_BIN)
        )
        expected = shlex.quote(self.config.version)
        ensure = shlex.quote(
            f'[ -x {OPENCODE_BIN} ] && [ "$({OPENCODE_BIN} --version)" = {expected} ] || ({script})'
        )
        guarded = f"mkdir -p {OPENCODE_DIR} && flock {OPENCODE_DIR}/install.lock sh -c {ensure}"
        install = await runtime.run(["sh", "-c", guarded], {})
        if install.exit_code != 0:
            raise RuntimeError(f"OpenCode install failed: {install.stderr.strip()[-500:]}")

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        _, prompt = self.resolve_prompt(trace.task.data)
        if not isinstance(prompt, str):
            raise ValueError("OpenCode requires a string task prompt")

        model = f"{PROVIDER}/{ctx.model}"
        permission = {
            "*": "allow",
            **{tool: "deny" for tool in self.config.disabled_tools or []},
        }
        config = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "enabled_providers": [PROVIDER],
            "model": model,
            "small_model": model,
            "permission": permission,
            "provider": {
                PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Verifiers interception",
                    "options": {
                        "baseURL": endpoint,
                        "apiKey": f"{{env:{KEY_VAR}}}",
                    },
                    "models": {ctx.model: {"name": ctx.model}},
                }
            },
            "mcp": {
                name: {"type": "remote", "url": url, "enabled": True}
                for name, url in mcp_urls.items()
            },
        }
        env = {
            **self.config.resolved_env,
            KEY_VAR: secret,
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1",
            "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
            "OPENCODE_DISABLE_CHANNEL_DB": "1",
            "OPENCODE_PURE": "1",
            "XDG_CONFIG_HOME": ".vf-opencode/config",
            "XDG_DATA_HOME": ".vf-opencode/data",
            "XDG_CACHE_HOME": ".vf-opencode/cache",
        }
        return await runtime.run_program(
            [
                OPENCODE_BIN,
                "run",
                "--model",
                model,
                "--dangerously-skip-permissions",
                "--",
                prompt,
            ],
            env,
        )
