from typing import (
    List,
    Dict,
    TypedDict,
    Annotated,
    Any,
    Optional,
)  # <-- 确保 'Optional' 已导入
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# 'TypedDict' 是一种特殊的 Python 字典，
# 我们可以用它来严格定义 Agent "记忆" 中必须包含哪些键。


class AgentState(TypedDict):
    """
    这是 Agent 的“短期记忆”或“草稿纸”。
    它在 LangGraph 的所有节点之间传递。
    """

    # --- 核心对话 ---
    # 'messages' 是一个特殊的列表。
    # 'add_messages' 的意思是，当一个节点返回 "messages" 时，
    # 它不会覆盖旧消息，而是会 *追加* 新消息到列表中。
    # 这就是我们实现聊天历史的方式。
    messages: Annotated[list[BaseMessage], add_messages]

    # ---  CRISP-DM 阶段 1：业务理解 ---
    task: str  # 用户的原始请求 (例如 "帮我分析客户流失")
    business_objective: str  # Agent 提炼的业务目标 (例如 "识别高风险客户")

    # --- CRISP-DM 阶段 2 & 3：数据理解与准备 ---
    # (注意：我们不再需要 dataset_path，因为沙箱会自己管理)
    data_summary: str  # 'df.info()' 和 'df.describe()' 的摘要

    # --- CRISP-DM 阶段 4：模型构建 ---
    # (我们也不再需要 model_path)

    # --- CRISP-DM 阶段 5 & 6：评估与部署 ---
    evaluation_metrics: Dict[str, Any]  # 存储 {'accuracy': 0.9, 'f1_score': 0.88}
    xai_report: str  # SHAP/LIME 分析的文本摘要
    # xai_images: List[str]   # (我们的 FastAPI 服务器尚不支持这个，暂时注释掉)
    compliance_report: str  # 'Fairlearn' 公平性审计的结果
    final_report: str  # 最终给用户的总结报告
    # --- ⬇️ 这是关键修复 ⬇️ ---
    # 一个临时字段，用于在 code_generator 和 code_executor 之间传递 ID
    current_tool_call_id: Optional[str]
    # 一个临时字段，用于在 reflection 和 router 之间传递决策
    next_node: Optional[str]
    # --- ⬆️ 修复结束 ⬆️ ---
