from langchain_core.messages import ToolMessage, AIMessage
from ..state import AgentState
from ...tools.mcp_client import execute_code_in_sandbox


def code_executor_node(state: AgentState) -> dict:
    """
    CRISP-DM 步骤 3/4: 执行代码
    调用 FastAPI/MCP 服务器来运行代码。
    """
    print("--- [节点 3: 代码执行器] ---")

    # 1. 从状态中获取最后一条 AI 消息（即代码）
    last_message = state["messages"][-1]

    # --- ⬇️ 这是关键修复 ⬇️ ---
    # 1. 检查状态中是否有我们保存的 ID
    tool_call_id = state.get("current_tool_call_id")

    if not tool_call_id:
        return {
            "messages": [
                ToolMessage(
                    content="[错误] 找不到 current_tool_call_id。上一步 'code_generator' 必须返回一个。",
                    tool_call_id="error_handler",
                )
            ]
        }

    # 2. 从 AIMessage 的 tool_calls 中提取代码
    # (这提供了额外的验证)
    if not last_message.tool_calls or last_message.tool_calls[0]["id"] != tool_call_id:
        return {
            "messages": [
                ToolMessage(
                    content="[错误] 状态中的 tool_call_id 与 AI 消息不匹配。",
                    tool_call_id=tool_call_id,
                )
            ]
        }

    code_to_run = last_message.tool_calls[0]["args"]["code_string"]
    # --- ⬆️ 修复结束 ⬆️ ---

    # 3. (关键) 调用我们的 MCP 客户端
    result_dict = execute_code_in_sandbox(code_to_run)

    result_string = result_dict.get("result", "没有收到来自沙箱的输出。")

    print(f"代码执行结果 (前 200 字符):\n{result_string[:200]}...")

    # 4. (关键) 返回带有 *正确* tool_call_id 的 ToolMessage
    return {"messages": [ToolMessage(content=result_string, tool_call_id=tool_call_id)]}
