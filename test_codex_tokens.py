from pathlib import Path
import json


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [
            "codex",
            "exec",
            "Create a simple python script that prints 'Hello, World!'",
            "--profile",
            "gpt-5-1-codex-max",
            "--skip-git-repo-check",
            "--yolo",
            "--json"
        ],
    )
    print("Result:")
    print("stdout:")
    print(result.stdout)
    print("stderr:")
    print(result.stderr)
    print("returncode:")
    print(result.returncode)
