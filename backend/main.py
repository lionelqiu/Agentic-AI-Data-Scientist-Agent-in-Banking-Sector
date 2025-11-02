import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import atexit

# ----------------------------------------------------------------------
# 1. (关键) 将 'src' 目录添加到 Python 路径
#    这允许 main.py 找到并导入 'src.bank_ds_agent...'
# ----------------------------------------------------------------------
# (os.path.dirname(__file__) 是 '.../backend')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

try:
    # ----------------------------------------------------------------------
    # 2. (关键) 导入我们刚刚造好的【轮子 2】
    # ----------------------------------------------------------------------
    from src.bank_ds_agent.tools.code_executor import SandboxJupyterExecutor
except ImportError as e:
    print(f"致命错误: 无法导入 SandboxJupyterExecutor。")
    print(f"请确保 __init__.py 文件存在，并且 'src' 在路径中: {e}")
    sys.exit(1)


# ----------------------------------------------------------------------
# 3. Pydantic 模型（我们的 MCP 消息格式）
# ----------------------------------------------------------------------
class CodeRequest(BaseModel):
    # session_id: str  # (未来用于区分不同用户的沙箱)
    code: str


class CodeResponse(BaseModel):
    result: str


# ----------------------------------------------------------------------
# 4. FastAPI 应用和“单例”执行器
# ----------------------------------------------------------------------
app = FastAPI(
    title="银行 Agent - Jupyter 沙箱工具服务器 (MCP)",
    description="在安全的 Docker 容器中执行有状态的 Python 代码。",
)

# 全局变量，用于持有我们的执行器“轮子”
executor: SandboxJupyterExecutor = None


@app.on_event("startup")
async def startup_event():
    """
    当 FastAPI 服务器启动时，自动构建 Docker 镜像
    并启动我们的沙箱执行器。
    """
    global executor
    print("FastAPI 正在启动...")
    try:
        # 步骤 1: 构建镜像 (这会使用缓存，除非 Dockerfile.agent 更改)
        from src.bank_ds_agent.tools.code_executor import build_docker_image

        print("正在构建/验证 Docker 镜像...")
        # (project_root 变量已在该文件的顶部定义)
        build_docker_image(
            image_tag="agent-executor:latest", build_context_path=project_root
        )  # <-- 修复！

        # 步骤 2: (关键) 初始化我们的沙箱
        print("正在启动 SandboxJupyterExecutor...")
        executor = SandboxJupyterExecutor(image_name="agent-executor:latest")
        print("FastAPI 启动成功：沙箱已准备就绪。")

    except Exception as e:
        print(f"!! 致命错误：FastAPI 启动失败 !!")
        print(f"!! 无法初始化沙箱: {e}")
        # (在生产中，这应该会使服务器崩溃并重启)
        executor = None


@app.on_event("shutdown")
async def shutdown_event():
    """
    当 FastAPI 服务器关闭时，清理内核和容器。
    """
    global executor
    if executor:
        print("FastAPI 正在关闭...")
        executor.cleanup()


# ----------------------------------------------------------------------
# 5. MCP API 端点
# ----------------------------------------------------------------------
@app.post("/execute", response_model=CodeResponse)
async def execute_code_endpoint(request: CodeRequest):
    """
    执行代码（有状态）
    """
    global executor
    if not executor:
        raise HTTPException(status_code=503, detail="沙箱服务不可用。")

    try:
        # (关键) 调用我们轮子的 .execute() 方法
        result_string = executor.execute(request.code)

        return CodeResponse(result=result_string)

    except Exception as e:
        # (这不应该发生，因为 execute() 已经捕获了错误)
        raise HTTPException(status_code=500, detail=f"执行时发生内部错误: {e}")


if __name__ == "__main__":
    # 允许直接运行此文件 (尽管我们更推荐 'uvicorn main:app')
    print("正在启动 Uvicorn (调试模式)...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
