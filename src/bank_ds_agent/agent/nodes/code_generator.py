# (文件顶部的 import 和提示词保持不变)
import re
import json
import time  # <-- 导入 time
import os  # <-- 导入 os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..state import AgentState
from ..llms import get_llm

CODE_GENERATOR_SYSTEM_PROMPT = """
你是一个专业的 Python 数据科学家。
你拥有一个名为 'PythonCode' 的工具，该工具只有一个参数 'code_string'。
你的任务是根据一个目标和对话历史，调用 'PythonCode' 工具来执行下一步。

规则:
1.  **必须**使用 'PythonCode' 工具来提交你的代码。
2.  你只能访问 (pandas, sklearn, matplotlib, shap, fairlearn, dill)。
3.  **不要**做任何 `pip install` 操作。
4.  你的代码应该是 *有状态的*。你可以假设在 /app/session.dill 中保存了之前的变量。
"""


def _parse_code_block(text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip().replace("```", "")


# --- ⬇️ 替换这个函数 ⬇️ ---
def code_generator_node(state: AgentState) -> dict:
    """
    CRISP-DM 步骤 3/4: 数据准备/模型构建
    调用 LLM 来生成一个 *工具调用* (Tool Call)。
    """
    print("--- [节点 2: 代码生成器] ---")

    llm = get_llm()

    messages_for_prompt = [SystemMessage(content=CODE_GENERATOR_SYSTEM_PROMPT)]
    prompt = f"业务目标: {state['business_objective']}\n\n"
    prompt += "根据这个目标和下面的历史记录，为下一步调用 PythonCode 工具：\n"
    prompt += "--- 历史记录 ---\n"
    for msg in state["messages"][-5:]:
        prompt += f"{msg.type}: {msg.content}\n"
    messages_for_prompt.append(HumanMessage(content=prompt))

    print(f"正在调用 '{os.getenv('LLM_BACKEND')}' LLM (以获取工具调用)...")

    tool_call_id = None
    response_message = None

    if hasattr(llm, "invoke"):
        # --- 这是 LangChain (API) 的方式 ---
        response_message = llm.invoke(messages_for_prompt)
        if not response_message.tool_calls:
            print("!! 错误: (API) LLM 未返回工具调用，返回了一个普通消息。")
            return {"messages": [response_message]}
        tool_call = response_message.tool_calls[0]
        code_string = tool_call["args"]["code_string"]
        tool_call_id = tool_call["id"]
    else:
        # --- ⬇️ 这是 Llama.cpp 的方式 (已修复) ⬇️ ---
        # (手动将 LangChain 消息转换为 Llama.cpp 字典)
        messages_as_dicts = []
        for msg in messages_for_prompt:
            if msg.type == "human":
                messages_as_dicts.append({"role": "user", "content": msg.content})
            elif msg.type == "system":
                messages_as_dicts.append({"role": "system", "content": msg.content})

        response = llm.create_chat_completion(
            messages=messages_as_dicts, temperature=0.0
        )
        response_content = response["choices"][0]["message"]["content"].strip()

        try:
            print(f"Llama.cpp 原始输出: {response_content}")
            code_string = _parse_code_block(response_content)
            tool_call_id = f"local_tool_call_{int(time.time())}"
            response_message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "PythonCode",
                        "args": {"code_string": code_string},
                    }
                ],
            )
            if not code_string:
                raise ValueError("Llama.cpp 未返回代码块。")

        except Exception as e:
            print(f"!! 错误: (Local) LLM 未返回可解析的代码。错误: {e}")
            return {"messages": [AIMessage(content=f"错误: {e}")]}
        # --- ⬆️ 修复结束 ⬆️ ---

    print(f"生成的代码:\n{code_string[:200]}...")
    print(f"生成的 Tool Call ID: {tool_call_id}")

    return {"messages": [response_message], "current_tool_call_id": tool_call_id}
