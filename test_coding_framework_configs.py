"""Quick test suite to test codex command with different model profiles."""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models import MODELS


def test_codex_command(model: str) -> Tuple[bool, int, str, str]:
    """
    Test codex command with a specific model profile.

    Args:
        model: Model name to use as profile

    Returns:
        Tuple of (success: bool, return_code: int, stdout: str, stderr: str)
    """

    if model == "claude-opus-4-5" or model == "claude-sonnet-4-5":
        command = [
            "claude",
            "-p",
            "Hello",
            "--model",
            model,
            "--dangerously-skip-permissions",
        ]
    elif model == "gpt-5-1-codex-max" or model == "gpt-5-2":
        command = [
            "codex",
            "exec",
            "Hello",
            "--profile",
            model,
            "--skip-git-repo-check",
            "--yolo",
        ]

    elif model.startswith("openrouter/"):
        command = [
            "opencode",
            "run",
            "Hello",
            "--model",
            model,
        ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
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
