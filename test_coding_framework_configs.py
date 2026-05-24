"""Quick test suite to test codex command with different model profiles."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models import (
    MODELS,
    OPENROUTER_CLAUDE_CODE_MODELS,
    OPENROUTER_CODEX_PROFILES,
)

CLAUDE_CODE_MODEL_ALIASES = {
    "claude-opus-4-5": "claude-opus-4-5",
    "claude-sonnet-4-5": "claude-sonnet-4-5",
}


def build_claude_openrouter_env() -> dict[str, str]:
    env = os.environ.copy()
    openrouter_api_key = env.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY must be set to test Claude Code through OpenRouter"
        )

    env["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
    env["ANTHROPIC_AUTH_TOKEN"] = openrouter_api_key
    env["ANTHROPIC_API_KEY"] = ""

    return env


def test_codex_command(model: str) -> Tuple[bool, int, str, str]:
    """
    Test codex command with a specific model profile.

    Args:
        model: Model name to use as profile

    Returns:
        Tuple of (success: bool, return_code: int, stdout: str, stderr: str)
    """

    env = os.environ.copy()

    if model in OPENROUTER_CLAUDE_CODE_MODELS:
        command = [
            "claude",
            "-p",
            "Hello",
            "--model",
            OPENROUTER_CLAUDE_CODE_MODELS[model],
            "--dangerously-skip-permissions",
        ]
        env = build_claude_openrouter_env()
    elif model in CLAUDE_CODE_MODEL_ALIASES:
        command = [
            "claude",
            "-p",
            "Hello",
            "--model",
            CLAUDE_CODE_MODEL_ALIASES[model],
            "--dangerously-skip-permissions",
        ]
    elif model.startswith("gpt"):
        command = [
            "codex",
            "exec",
            "Hello",
            "--profile",
            model,
            "--skip-git-repo-check",
            "--yolo",
        ]

    elif model in OPENROUTER_CODEX_PROFILES:
        command = [
            "codex",
            "exec",
            "Hello",
            "--profile",
            OPENROUTER_CODEX_PROFILES[model],
            "--skip-git-repo-check",
            "--yolo",
        ]

    # elif model.startswith("openrouter/"):
    #     command = [
    #         "opencode",
    #         "run",
    #         "Hello",
    #         "--model",
    #         model,
    #     ]
    else:
        return False, 1, "", f"No configured test command for model: {model}"

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        print(result.stdout)
        success = result.returncode == 0
        return success, result.returncode, result.stdout, result.stderr

    except Exception as e:
        return False, 1, "", str(e)


if __name__ == "__main__":
    print(f"Testing {len(MODELS)} models...\n")

    successful: List[str] = []
    failed: Dict[str, Tuple[int, str, str]] = {}

    for model in tqdm(MODELS, desc="Testing models", unit="model", ncols=100):
        tqdm.write(f"Testing: {model}")
        success, return_code, stdout, stderr = test_codex_command(model)
        if success:
            successful.append(model)
            tqdm.write(f"  ✅ {model} - SUCCESS")
        else:
            failed[model] = (return_code, stdout, stderr)
            tqdm.write(f"  ❌ {model} - FAILED (code: {return_code})")

    # Print summary
    print(f"{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}\n")

    if successful:
        print(f"✅ SUCCESS ({len(successful)}/{len(MODELS)}):")
        for model in successful:
            print(f"   - {model}")
        print()

    if failed:
        print(f"❌ FAILED ({len(failed)}/{len(MODELS)}):")
        for model, (return_code, stdout, stderr) in failed.items():
            print(f"\n   Model: {model}")
            print(f"   Return code: {return_code}")
            if stdout.strip():
                print(f"   STDOUT: {stdout.strip()[:200]}...")
            if stderr.strip():
                # Extract error message from stderr
                error_lines = [
                    line
                    for line in stderr.split("\n")
                    if "ERROR" in line or "error" in line.lower()
                ]
                if error_lines:
                    print(f"   ERROR: {error_lines[-1]}")
                else:
                    print(f"   STDERR: {stderr.strip()[:200]}...")
        print()

    print(f"{'=' * 60}")
    print(f"Total: {len(successful)} successful, {len(failed)} failed")
    print(f"{'=' * 60}")
