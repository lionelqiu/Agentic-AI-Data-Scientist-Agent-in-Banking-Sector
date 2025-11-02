from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from ..state import AgentState
from ..llms import get_llm
import os  # <-- 确保导入 os

# --- ⬇️ 适用于 8B 模型的“更简单”的提示词 ⬇️ ---
REFLECTION_SYSTEM_PROMPT = """
你是一个代码审查员。
你将看到一个“目标”和一个刚刚执行的“代码输出”。
判断这个“目标”是否已经完成。
只回答 "complete" 或 "continue"。

目标: {objective}
代码输出:
{output}

你的回答 (complete/continue):
"""
# --- ⬆️ 提示词结束 ⬆️ ---


def reflection_node(state: AgentState) -> dict:
    """
    CRISP-DM 步骤 5: 评估
    评估上一步代码执行的结果，并决定下一步的路由。
    """
    print("--- [节点 4: 反思] ---")

    # (安全网: 如果规划师失败了，就直接结束循环)
    if not state.get("business_objective") or state["business_objective"].strip() == "":
        print("!! 警告: 业务目标为空。强制结束循环。")
        return {"next_node": "complete"}  # <-- 修复 1 (将更新状态)

    last_message = state["messages"][-1]
    if not isinstance(last_message, ToolMessage):
        raise ValueError("反思节点的上一步必须是 ToolMessage")

    tool_output = last_message.content

    if (
        "[Error]" in tool_output
        or "[MCP 致命错误]" in tool_output
        or "Stderr:" in tool_output
    ):
        print("检测到代码执行错误。")
        return {
            "messages": [
                HumanMessage(
                    content="你的上一步代码执行失败了。请仔细检查错误并修复它。"
                )
            ],
            "next_node": "continue",  # <-- 修复 2 (将更新状态)
        }

    llm = get_llm()

    print(f"代码执行成功。正在调用 '{os.getenv('LLM_BACKEND')}' LLM 进行评估...")

    prompt = REFLECTION_SYSTEM_PROMPT.format(
        objective=state["business_objective"], output=tool_output
    )

    messages = [HumanMessage(content=prompt)]

    decision = ""
    if hasattr(llm, "invoke"):
        # --- 这是 LangChain (API) 的方式 ---
        response = llm.invoke(messages)
        if isinstance(response.content, list):
            for part in response.content:
                if isinstance(part, str):
                    decision = part
                    break
                elif isinstance(part, dict) and "text" in part:
                    decision = part["text"]
                    break
        else:
            decision = response.content
    else:
        # --- 这是 Llama.cpp 的方式 ---
        messages_as_dicts = []
        for msg in messages:
            if msg.type == "human":
                messages_as_dicts.append({"role": "user", "content": msg.content})
            elif msg.type == "system":
                messages_as_dicts.append({"role": "system", "content": msg.content})

        response = llm.create_chat_completion(
            messages=messages_as_dicts, temperature=0.0
        )
        decision = response["choices"][0]["message"]["content"].strip()

    decision = decision.strip().lower()
    print(f"“{os.getenv('LLM_BACKEND')}”的决定是: {decision}")

    # --- ⬇️ 这是关键修复 ⬇️ ---
    # 我们返回一个字典，LangGraph 会自动用它来更新 AgentState
    if "complete" in decision:
        print("--- [反思] 决策：任务已完成。---")
        return {"next_node": "complete"}

    # 否则 (如果它说 "continue" 或任何其他垃圾信息)，我们就继续
    print("--- [反思] 决策：任务继续。---")
    return {"next_node": "continue"}
    # --- ⬆️ 修复结束 ⬆️ ---
