import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
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
    def __init__(
        self,
        volume_path: str,
        run_hash: str,
        image_tag: str,
    ):
        """Initialize a Docker sandbox instance ready to run code.

        Args:
            volume_path: Host volume path to mount inside the container.
            run_hash: Identifier for the current sandbox run.
            image_tag: Docker image tag to use or build for this sandbox.

        Returns:
            None: Sets up Docker client references for sandbox operations.
        """

        self.client = docker.from_env()
        self.container = None
        self.volume = volume_path
        self.run_hash = run_hash
        self.image_tag = image_tag

    def create_container(
        self,
        additional_env: Optional[Dict[str, str]] = None,
    ):
        tag = self.image_tag

        try:
            self.client.images.get(tag)
        except docker.errors.ImageNotFound:
            try:
                self.client.images.build(
                    path=".",
                    tag=tag,
                    rm=True,
                    forcerm=True,
                    buildargs={},
                )
            except docker.errors.BuildError as e:
                print("Build error logs:")
                for log in e.build_log:
                    if "stream" in log:
                        print(log["stream"].strip())
                raise

        extra_hosts = {"host.docker.internal": "host-gateway"}
        environment = {
            "PHEONIX_COLLECTOR_ENDPOINT": "http://host.docker.internal:6006/v1/traces"
        }

        if additional_env:
            environment.update(additional_env)

        self.container = self.client.containers.run(
            tag,
            command="tail -f /dev/null",
            detach=True,
            tty=True,
            mem_limit="100gb",
            cap_drop=["ALL"],
            environment=environment,
            working_dir="/workspace",
            extra_hosts=extra_hosts
        )

    def _copy_directory_to_container(self, source: Path, destination: Path) -> None:
        """Copy ``source`` directory from host into container ``destination`` path.

        Args:
            source (Path): Host directory to copy.
            destination (Path): Target path inside the container.

        Returns:
            None: Copies data into the running container.
        """
        if not source.exists() or not source.is_dir():
            return

        archive_stream = io.BytesIO()
        with tarfile.open(fileobj=archive_stream, mode="w") as tar:
            tar.add(str(source), arcname=destination.name)

        archive_stream.seek(0)
        self.container.exec_run(["mkdir", "-p", str(destination.parent)])
        self.container.exec_run(["rm", "-rf", str(destination)])
        self.container.put_archive(str(destination.parent), archive_stream.getvalue())

    def _copy_directory_from_container(self, source: Path, destination: Path) -> None:
        """Copy ``source`` directory from container into host ``destination`` path.

        Args:
            source (Path): Directory inside the container.
            destination (Path): Destination directory on the host.

        Returns:
            None: Extracts container data into the host filesystem.
        """
        try:
            stream, _ = self.container.get_archive(str(source))
        except docker.errors.NotFound:
            return

        temp_dir = Path(tempfile.mkdtemp())
        try:
            archive_stream = io.BytesIO()
            for chunk in stream:
                archive_stream.write(chunk)
            archive_stream.seek(0)

            with tarfile.open(fileobj=archive_stream, mode="r") as tar:
                tar.extractall(path=temp_dir)

            extracted = temp_dir / source.name
            destination = Path(destination).resolve()
            if destination.exists():
                shutil.rmtree(destination)
            if extracted.is_dir():
                shutil.copytree(extracted, destination)
            elif extracted.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(extracted, destination)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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
        if envs and "TASK_ID" in envs and envs["TASK_ID"]:
            self.image_tag = envs["TASK_ID"]

        if not self.container:
            self.create_container(
                additional_env=envs,
            )

        output_root = Path(output_path) if output_path else None
        results_root = Path(results_path) if results_path else None
        inputs_root = Path(task_data_path) if task_data_path else None

        if inputs_root:
            inputs_root = inputs_root.resolve()
            self._copy_directory_to_container(inputs_root / "data", Path("/workspace/data"))
            self._copy_directory_to_container(inputs_root / "reference", Path("/workspace/reference"))

        log_dir_candidates = [root.parent for root in (output_root, results_root, inputs_root) if root is not None]
        log_dir = log_dir_candidates[0] if log_dir_candidates else Path.cwd()
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "docker_stdout.log"

        try:
            exec_result = self.container.exec_run(
                cmd=["/opt/conda/bin/python", "-c", code],
                user="root",
                environment=envs,
                stream=True,
                demux=True,
            )

            output_chunks: List[str] = []
            with log_file_path.open("w", encoding="utf-8") as log_file:
                for stdout_chunk, stderr_chunk in exec_result.output:
                    if stdout_chunk:
                        text = stdout_chunk.decode()
                        output_chunks.append(text)
                        log_file.write(text)
                        log_file.flush()
                    if stderr_chunk:
                        text = stderr_chunk.decode()
                        output_chunks.append(text)
                        log_file.write(text)
                        log_file.flush()

            output = "".join(output_chunks)
            exit_code = exec_result.exit_code

            if exit_code != 0:
                error = ExecutionError(traceback=output)
                result = ExecutionResult(output=output, error=error)
            else:
                result = ExecutionResult(output=output)

        except Exception as e:
            error = ExecutionError(traceback=str(e))
            result = ExecutionResult(error=error)
        finally:
            if output_root:
                output_root.mkdir(parents=True, exist_ok=True)
                self._copy_directory_from_container(Path("/workspace/output"), output_root)
            if results_root:
                results_root.mkdir(parents=True, exist_ok=True)
                self._copy_directory_from_container(Path("/workspace/results"), results_root)

        return result


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