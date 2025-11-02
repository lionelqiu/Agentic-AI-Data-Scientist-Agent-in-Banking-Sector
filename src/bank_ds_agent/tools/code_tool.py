from pydantic import BaseModel, Field


class PythonCode(BaseModel):
    """
    一个 Pydantic 模型，用于定义我们的 'run_python_code' 工具。
    LLM 将被训练来填充这个模型的字段。
    """

    code_string: str = Field(
        ...,
        description="要在一个有状态的、沙箱化的 Jupyter 内核中执行的、格式正确的 Python 代码字符串。",
    )
