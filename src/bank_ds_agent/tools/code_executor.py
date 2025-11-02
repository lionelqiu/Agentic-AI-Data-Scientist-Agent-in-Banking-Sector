import docker
import jupyter_client
import time
import os
import json
import tempfile
import shutil
import atexit
import re  # <-- 确保 re 被导入
from queue import Empty
import docker.errors  # <-- 确保 docker.errors 被导入


class SandboxJupyterExecutor:
    """
    一个有状态的、沙箱化的 Jupyter 执行器。（来自您的优秀参考）

    它通过 Docker 启动一个 Jupyter 内核容器，并使用 jupyter_client
    通过 TCP 端口（9000-9004）连接到它。
    """

    def __init__(self, image_name="agent-executor:latest", timeout=20):
        print(f"Initializing SandboxJupyterExecutor with image {image_name}...")
        self.client = docker.from_env()
        self.image_name = image_name
        self.container = None
        self.km = None

        # 解决方案 1：使用临时目录进行卷挂载
        self.kernel_dir = tempfile.mkdtemp(prefix="agent_kernel_")
        self.kernel_json_path = os.path.join(self.kernel_dir, "kernel.json")

        # 端口必须与 Dockerfile.agent 中的 CMD 匹配
        self.ports = {f"{p}/tcp": p for p in range(9000, 9005)}

        try:
            # 启动容器
            print(f"Starting container from image {self.image_name}...")
            self.container = self.client.containers.run(
                image=self.image_name,
                detach=True,
                ports=self.ports,
                volumes={self.kernel_dir: {"bind": "/app", "mode": "rw"}},
                auto_remove=False,  # 我们将在 cleanup() 中手动删除
                publish_all_ports=False,
            )

            # 解决方案 2：等待 kernel.json 文件出现
            print(f"Waiting for kernel.json to appear at {self.kernel_json_path}...")
            start_time = time.time()
            while not os.path.exists(self.kernel_json_path):
                if time.time() - start_time > timeout:
                    raise TimeoutError("Kernel failed to start and write kernel.json")
                time.sleep(0.1)

                # 检查容器是否意外退出
                self.container.reload()
                if self.container.status == "exited":
                    logs = self.container.logs().decode("utf-8")
                    raise RuntimeError(f"Container exited unexpectedly. Logs:\n{logs}")

            print("kernel.json found. Patching IP address...")

            # 解决方案 3：修补 kernel.json
            with open(self.kernel_json_path, "r+") as f:
                config = json.load(f)
                config["ip"] = "127.0.0.1"
                f.seek(0)
                json.dump(config, f)
                f.truncate()

            print("Connecting jupyter_client...")
            self.km = jupyter_client.BlockingKernelClient()
            self.km.load_connection_file(self.kernel_json_path)
            self.km.start_channels()

            # 解决方案 4：健壮的连接握手
            try:
                print("Testing kernel connection (wait_for_ready)...")
                self.km.wait_for_ready(timeout=timeout)
                print("Kernel is alive and ready!")
            except RuntimeError as e:
                print(f"Kernel connection test failed: {e}")
                print("This is often a FIREWALL or ANTIVIRUS issue.")
                print("---!!!--- Retrieving container logs for debugging ---!!!---")
                try:
                    self.container.reload()
                    logs = self.container.logs().decode("utf-8")
                    print(f"Container '{self.container.short_id}' logs:\n{logs}")
                except Exception as log_e:
                    print(f"Failed to retrieve container logs: {log_e}")
                self.cleanup()
                raise

            atexit.register(self.cleanup)

        except Exception as e:
            print(f"Error during initialization: {e}")
            if self.container:
                print("---!!!--- Retrieving container logs for debugging ---!!!---")
                try:
                    self.container.reload()
                    logs = self.container.logs().decode("utf-8")
                    print(f"Container '{self.container.short_id}' logs:\n{logs}")
                except Exception as log_e:
                    print(f"Failed to retrieve container logs: {log_e}")
            self.cleanup()  # 确保在失败时清理
            raise

    def execute(self, code, timeout=10):
        """
        在沙箱化、有状态的内核中执行代码。
        """
        if not self.km:
            raise RuntimeError("Executor is not initialized or has been cleaned up.")

        print(f"\n[Executing Code]:\n{code}\n")
        msg_id = self.km.execute(code)
        outputs = []

        try:
            # 2. 等待 shell 通道的最终执行回复
            reply = self.km.get_shell_msg(timeout=timeout)

            if reply["content"]["status"] == "error":
                error_content = reply["content"]
                traceback = "\n".join(error_content.get("traceback", []))
                # (清理 ANSI 颜色代码)
                traceback = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", traceback)
                outputs.append(
                    f"[Error] {error_content.get('ename', 'UnknownError')}: {error_content.get('evalue', '')}\n{traceback}"
                )

        except Empty:
            return f"[Error] Timeout: Code execution took too long (> {timeout}s)."
        except Exception as e:
            return f"[Error] Failed to get shell reply: {e}"

        # 3. 排空 IOPub 通道
        while True:
            try:
                msg = self.km.get_iopub_msg(timeout=0.2)
            except Empty:
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["header"]["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                outputs.append(f"[{content['name']}] {content['text']}")
            elif msg_type == "display_data":
                outputs.append(
                    f"[Display] {content['data'].get('text/plain', 'No plain text representation')}"
                )
            elif msg_type == "execute_result":
                outputs.append(
                    f"[Result] {content['data'].get('text/plain', 'No plain text representation')}"
                )
            elif msg_type == "error":
                traceback = "\n".join(content.get("traceback", []))
                traceback = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", traceback)
                outputs.append(
                    f"[Error] {content.get('ename', 'UnknownError')}: {content.get('evalue', '')}\n{traceback}"
                )

        result = "\n".join(outputs)
        print(f"[Execution Result]:\n{result}")
        return result

    def cleanup(self):
        """
        停止内核、停止容器并删除临时目录。
        """
        print("\nCleaning up resources...")
        try:
            if self.km and self.km.is_alive():
                print("Shutting down kernel...")
                self.km.shutdown()
        except Exception as e:
            print(f"Error shutting down kernel: {e}")

        try:
            if self.container:
                print(f"Stopping container {self.container.short_id}...")
                self.container.stop()
                print("Container stopped.")
                print(f"Removing container {self.container.short_id}...")
                self.container.remove()
                print("Container removed.")
        except docker.errors.NotFound:
            print("Container already stopped or removed.")
        except Exception as e:
            print(f"Error stopping/removing container: {e}")

        try:
            if self.kernel_dir and os.path.exists(self.kernel_dir):
                print(f"Removing temp directory {self.kernel_dir}...")
                shutil.rmtree(self.kernel_dir)
                print("Temp directory removed.")
        except Exception as e:
            print(f"Error removing temp directory: {e}")

        self.km = None
        self.container = None


def build_docker_image(
    image_tag="agent-executor:latest", build_context_path="."
):  # <-- 1. 添加参数
    """
    自动构建 Docker 镜像。
    """
    print(f"Building Docker image '{image_tag}' from Dockerfile.agent...")
    try:
        client = docker.from_env()
        image, logs = client.images.build(
            path=build_context_path,  # <-- 2. 使用该参数
            dockerfile="Dockerfile.agent",
            tag=image_tag,
            rm=True,
        )
        print("Docker image built successfully.")
        return image
    except Exception as e:
        print(f"Failed to build Docker image: {e}")
        print(
            "Please ensure Docker is running and Dockerfile.agent is in the same directory."
        )
        raise
