import requests
import json

# 这是我们 FastAPI/MCP 服务器的地址
TOOL_SERVER_URL = "http://127.0.0.1:8000"


def execute_code_in_sandbox(code: str) -> dict:
    """
    调用我们的 FastAPI/MCP 服务器来执行代码。
    这是 Agent 的“双手”。
    """
    print(f"--- [MCP 客户端] 正在向沙箱发送代码 ---")
    try:
        response = requests.post(
            f"{TOOL_SERVER_URL}/execute",
            json={"code": code},
            timeout=60,  # 增加超时以应对长时间运行的代码
        )

        if response.status_code == 200:
            # 成功
            return response.json()
        else:
            # API 服务器返回了一个 HTTP 错误
            return {
                "result": f"[MCP 错误] 服务器返回状态 {response.status_code}: {response.text}"
            }

    except requests.exceptions.ConnectionError:
        return {
            "result": "[MCP 致命错误] 无法连接到沙箱服务器 (FastAPI)。"
            "请确保 'backend/main.py' 正在运行。"
        }
    except Exception as e:
        return {"result": f"[MCP 致命错误] 发生意外错误: {e}"}
