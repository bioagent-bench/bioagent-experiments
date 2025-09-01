import os
from typing import Optional, Dict, Any, List

import docker


class ExecutionError:
    def __init__(self, traceback: str):
        self.traceback = traceback


class ExecutionLogs:
    def __init__(self, stdout: List[str]):
        self.stdout = stdout


class ExecutionResult:
    def __init__(self, output: str = "", error: Optional[ExecutionError] = None):
        self.output = output
        self.error = error
        self.logs = ExecutionLogs([output] if output else [])


class DockerSandbox:
    def __init__(self, volume_path: Optional[str] = None):
        self.client = docker.from_env()
        self.container = None
        self.volume = volume_path

    def create_container(self, host_path: Optional[str] = None, container_path: Optional[str] = "/workspace"):
        try:
            image, build_logs = self.client.images.build(
                path=".",
                tag="agent-sandbox",
                rm=True,
                forcerm=True,
                buildargs={},
            )
        except docker.errors.BuildError as e:
            print("Build error logs:")
            for log in e.build_log:
                if 'stream' in log:
                    print(log['stream'].strip())
            raise

        # Prepare volume mounts
        volumes = {}
        if host_path and container_path:
            volumes[host_path] = {
                'bind': container_path,
                'mode': 'rw'
            }

        self.container = self.client.containers.run(
            "agent-sandbox",
            command="tail -f /dev/null",
            detach=True,
            tty=True,
            mem_limit="100gb",
            cap_drop=["ALL"],
            volumes=volumes,
            environment={
                "HF_TOKEN": os.getenv("HF_TOKEN")
            },
        )

    def run_code(self, code: str, host_path: Optional[str] = None, container_path: Optional[str] = "/workspace", envs: Optional[Dict[str, str]] = None) -> ExecutionResult:
        if not self.container:
            self.create_container(host_path=host_path, container_path=container_path)

        try:
            exec_result = self.container.exec_run(
                cmd=["python", "-c", code],
                user="nobody",
                environment=envs
            )

            output = exec_result.output.decode() if exec_result.output else ""
            exit_code = exec_result.exit_code

            if exit_code != 0:
                error = ExecutionError(traceback=output)
                return ExecutionResult(output=output, error=error)
            else:
                return ExecutionResult(output=output)

        except Exception as e:
            error = ExecutionError(traceback=str(e))
            return ExecutionResult(error=error)


    def cleanup(self):
        if self.container:
            try:
                self.container.stop()
            except docker.errors.NotFound:
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                self.container = None