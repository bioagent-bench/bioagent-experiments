import os
from typing import Optional, Dict, Any, List

import docker


class DockerSandbox:
    def __init__(self):
        self.client = docker.from_env()
        self.container = None

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
            mem_limit="512m",
            cpu_quota=50000,
            pids_limit=100,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            volumes=volumes,
            environment={
                "HF_TOKEN": os.getenv("HF_TOKEN")
            },
        )

    def run_code(self, code: str, host_path: Optional[str] = None, container_path: Optional[str] = "/workspace") -> Optional[str]:
        if not self.container:
            self.create_container(host_path=host_path, container_path=container_path)

        exec_result = self.container.exec_run(
            cmd=["python", "-c", code],
            user="nobody"
        )
        return exec_result.output.decode() if exec_result.output else None


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