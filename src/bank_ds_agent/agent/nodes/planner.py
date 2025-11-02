import os
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState
from ..llms import get_llm, unload_llms

# --- ⬇️ 适用于 8B 模型的“更简单”的提示词 ⬇️ ---
PLANNER_SYSTEM_PROMPT = """
你是一个代码助手。
用户的任务在下面。请将这个任务*总结*成一个单一句子的目标。
只返回这个句子。

任务:
{task}

总结的目标:
"""
# --- ⬆️ 提示词结束 ⬆️ ---


def planner_node(state: AgentState) -> dict:
    print("--- [节点 1: 规划师] ---")

    llm = get_llm()
    prompt = PLANNER_SYSTEM_PROMPT.format(task=state["task"])

    messages = [
        # (对于 llama.cpp，最好只使用 HumanMessage)
        HumanMessage(content=prompt)
    ]

    print(f"正在调用 '{os.getenv('LLM_BACKEND')}' LLM 进行规划...")

    if hasattr(llm, "invoke"):
        # --- 这是 LangChain (API) 的方式 ---
        response = llm.invoke(messages)
        business_objective = response.content.strip()
    else:
        # --- ⬇️ 这是 Llama.cpp 的方式 (已修复) ⬇️ ---
        # (手动将 LangChain 消息转换为 Llama.cpp 字典)
        messages_as_dicts = []
        for msg in messages:
            if msg.type == "human":
                messages_as_dicts.append({"role": "user", "content": msg.content})
            elif msg.type == "system":
                messages_as_dicts.append({"role": "system", "content": msg.content})

        response = llm.create_chat_completion(
            messages=messages_as_dicts, temperature=0.0
        )
        business_objective = response["choices"][0]["message"]["content"].strip()
        # --- ⬆️ 修复结束 ⬆️ ---

    print(f"提炼的目标: {business_objective}")

    return {
        "business_objective": business_objective,
        "messages": [HumanMessage(content=f"**目标已设定：** {business_objective}")],
    }
