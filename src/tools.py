from smolagents import tool


@tool
def run_terminal_command(command: str) -> str:
    """
    Run a terminal command and return combined stdout and stderr output.

    Args:
        command (str): Command to run in the shell.

    Returns:
        str: Combined stdout and stderr output, or an error description.
    """
    import subprocess

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        combined_output = "\n".join(part for part in (stdout, stderr) if part)
        body = combined_output if combined_output else "<no output>"
        return f"Exit code {result.returncode}\n{body}"
    except Exception as error:  # pragma: no cover - defensive
        return f"Unexpected error while running command: {error}"