import subprocess
from typing import Dict, Any
from smolagents import tool

@tool
def run_terminal_command(command: str) -> Dict[str, Any]:
    """
    Run a terminal command and return stdout, stderr, and exit code.

    Args:
        command (str): Command to run in the shell.

    Returns:
        Dict[str, Any]: Contains 'stdout', 'stderr', and 'exit_code'.
    """
    try:
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
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1
        }