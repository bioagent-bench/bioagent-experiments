import os
from typing import Optional, Dict, Any, List
from pathlib import Path

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
    def __init__(self, volume_path: Optional[str] = None, run_hash: Optional[str] = None):
        self.client = docker.from_env()
        self.container = None
        self.volume = volume_path
        self.run_hash = run_hash

    def create_container(
        self,
        task_data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        results_path: Optional[str] = None,
        additional_env: Optional[Dict[str, str]] = None,
    ):
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
        
        # Mount task data directory if provided
        if task_data_path:
            task_path = Path(task_data_path).resolve()
            
            # Mount data directory
            data_path = task_path / "data"
            if data_path.exists():
                volumes[str(data_path)] = {
                    'bind': '/workspace/data',
                    'mode': 'ro'  # Read-only for input data
                }
            
            # Mount reference directory
            reference_path = task_path / "reference"
            if reference_path.exists():
                volumes[str(reference_path)] = {
                    'bind': '/workspace/reference',
                    'mode': 'ro'  # Read-only for reference data
                }
        
        # Mount output directory if provided
        if output_path:
            # Ensure output path is absolute and exists with proper permissions
            output_abs_path = str(Path(output_path).resolve())
            output_resolved = Path(output_path)
            output_resolved.mkdir(parents=True, exist_ok=True)
            # Set permissions to allow writing from container
            os.chmod(output_resolved, 0o777)
            volumes[output_abs_path] = {
                'bind': '/workspace/output',
                'mode': 'rw'  # Read-write for output
            }

        # Mount results directory if provided
        if results_path:
            results_abs_path = str(Path(results_path).resolve())
            results_resolved = Path(results_path)
            results_resolved.mkdir(parents=True, exist_ok=True)
            os.chmod(results_resolved, 0o777)
            volumes[results_abs_path] = {
                'bind': '/workspace/results',
                'mode': 'rw'
            }

        extra_hosts = {"host.docker.internal": "host-gateway"}
        environment = {
            "PHEONIX_COLLECTOR_ENDPOINT": "http://host.docker.internal:6006/v1/traces"
        }

        if additional_env:
            environment.update(additional_env)

        self.container = self.client.containers.run(
            "agent-sandbox",
            command="tail -f /dev/null",
            detach=True,
            tty=True,
            mem_limit="100gb",
            cap_drop=["ALL"],
            volumes=volumes,
            environment=environment,
            working_dir="/workspace",
            extra_hosts=extra_hosts
        )

    def run_code(
        self,
        code: str,
        task_data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        results_path: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        run_hash: Optional[str] = None,
    ) -> ExecutionResult:
        if run_hash:
            self.run_hash = run_hash

        if not self.container:
            self.create_container(
                task_data_path=task_data_path,
                output_path=output_path,
                results_path=results_path,
                additional_env=envs,
            )

        try:
            exec_result = self.container.exec_run(
                cmd=["/opt/conda/bin/python", "-c", code],
                user="root",  # Use root to avoid permission issues
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