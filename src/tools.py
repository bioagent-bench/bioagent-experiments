from typing import Any
from smolagents import tool

@tool
def run_terminal_command(command: str) -> dict:
    """
    Run a terminal command and return stdout, stderr, and exit code.

    Args:
        command (str): Command to run in the shell.

    Returns:
        Dict[str, Any]: Contains 'stdout', 'stderr', and 'exit_code'.
    """
    import subprocess

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    return {
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "exit_code": result.returncode
    }