import docker
import jupyter_client
import time
import os
import atexit
import base64
from queue import Empty


class SandboxJupyterExecutor:
    """
    一个使用 Docker 容器作为安全沙箱，并在容器内运行 Jupyter 内核的工具。
    它保持了状态，同时提供了安全隔离。
    """

    def __init__(self, image_name="bank-agent-kernel"):
        print("正在初始化 Docker 沙箱...")
        self.client = docker.from_env()
        self.image_name = image_name
        self.container = None
        self.kernel_client = None
        self.connection_file_path = "temp_kernel.json"

        # 启动容器和内核
        self._start_container_and_kernel()

        # 确保在程序退出时清理容器
        atexit.register(self.close)
        print("Docker 沙箱内核已连接并准备就绪。")

    def _start_container_and_kernel(self):
        try:
            # 1. 启动 Docker 容器
            print(f"正在从镜像启动容器: {self.image_name}")
            self.container = self.client.containers.run(
                self.image_name,
                detach=True,  # 后台运行
                ports={"8888/tcp": 8888},  # (可选) 映射一个端口
                # (关键) 将容器内的 /app 目录挂载到本地，以便获取 kernel.json
                volumes={os.path.abspath("."): {"bind": "/app", "mode": "rw"}},
            )

            # 2. 等待 kernel.json 文件被创建
            # (容器内的 CMD 会创建 /app/kernel.json，它会出现在我们的本地目录中)
            print("等待内核启动并创建 kernel.json...")
            while not os.path.exists(self.connection_file_path):
                time.sleep(0.1)
                if self.container.status == "exited":
                    raise Exception("容器意外退出，请检查 Docker 日志。")

            print("kernel.json 已找到。")

            # 3. (关键) 修改连接文件以正确定位
            # kernel.json 内部的 IP 是容器的 IP (例如 172.x.x.x)，
            # 我们需要将其替换为 'localhost'，因为 Docker Desktop 会为我们映射端口。
            # (注意：这是一个简化的实现。在生产中，这需要更复杂的网络处理)
            # **更新：** 更简单的方法是让 jupyter_client 帮我们处理。

            # 4. 加载连接文件并启动客户端
            self.kernel_client = jupyter_client.BlockingKernelClient(
                connection_file=self.connection_file_path
            )
            self.kernel_client.load_connection_file()

            # (这是 Docker 网络的魔法)
            # 我们告诉客户端，内核的 IP 不是文件里写的那个 (容器IP)
            # 而是 'localhost' (127.0.0.1)
            self.kernel_client.ip = "127.0.0.1"

            self.kernel_client.start_channels()

            # 测试连接
            self.kernel_client.wait_for_ready(timeout=60)
            print("内核连接测试通过。")

        except Exception as e:
            print(f"启动沙箱时出错: {e}")
            self.close()  # 失败时清理
            raise

    def run_code(self, code_string: str):
        """在沙箱内核中执行代码并返回所有输出。"""
        if not self.kernel_client:
            raise Exception("内核未连接。")

        print(
            f"\n--- [沙箱Jupyter] 正在执行: ---\n{code_string}\n-----------------------------"
        )

        msg_id = self.kernel_client.execute(code_string)

        outputs = {"stdout": "", "stderr": "", "images": []}  # 存储 base64 编码的图片

        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=5)  # 增加超时

                if msg["parent_header"].get("msg_id") == msg_id:
                    msg_type = msg["header"]["msg_type"]

                    if msg_type == "stream":
                        if msg["content"]["name"] == "stdout":
                            outputs["stdout"] += msg["content"]["text"]
                        else:
                            outputs["stderr"] += msg["content"]["text"]

                    elif msg_type == "display_data":
                        if "image/png" in msg["content"]["data"]:
                            img_b64 = msg["content"]["data"]["image/png"]
                            outputs["images"].append(img_b64)

                    elif msg_type == "error":
                        outputs[
                            "stderr"
                        ] += f"{msg['content']['ename']}: {msg['content']['evalue']}\n"

                    elif msg_type == "execute_reply":
                        if msg["content"]["status"] == "error":
                            outputs[
                                "stderr"
                            ] += f"{msg['content']['ename']}: {msg['content']['evalue']}\n"
                        break

            except Empty:
                print("[沙箱Jupyter] 消息通道超时，假定执行完毕。")
                break

        print(f"[沙箱Jupyter] Stdout: {outputs['stdout'][:100]}...")
        print(f"[沙箱Jupyter] Stderr: {outputs['stderr']}")
        print(f"[沙箱Jupyter] 捕获图像: {len(outputs['images'])}")
        return outputs

    def close(self):
        """关闭客户端、内核并销毁 Docker 容器"""
        print("\n--- [沙箱Jupyter] 正在关闭 ---")

        # 1. 清理本地连接文件
        if os.path.exists(self.connection_file_path):
            os.remove(self.connection_file_path)

        # 2. 关闭客户端
        if self.kernel_client:
            self.kernel_client.stop_channels()
            self.kernel_client = None

        # 3. 停止并移除 Docker 容器
        if self.container:
            print(f"正在停止和移除容器: {self.container.id[:12]}...")
            try:
                self.container.stop()
                self.container.remove()
                print("容器已销毁。")
            except Exception as e:
                print(f"关闭容器时出错: {e}")  # (可能它已经停止了)
            self.container = None
