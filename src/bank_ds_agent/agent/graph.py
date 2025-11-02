from langgraph.graph import StateGraph, END  # <--- 修复 1
from .state import AgentState
from .nodes.planner import planner_node
from .nodes.code_generator import code_generator_node
from .nodes.code_executor import code_executor_node
from .nodes.reflection import reflection_node


def create_agent_graph():
    """
    创建并编译 agentic 循环图。
    """
    print("--- [Graph] 正在构建 Agent 状态机... ---")

    # 1. 初始化图，并绑定我们的“记忆”（AgentState）
    workflow = StateGraph(AgentState)  # <--- 修复 2

    # 2. 添加我们所有的“功能模块”（节点）
    workflow.add_node("planner", planner_node)
    workflow.add_node("code_generator", code_generator_node)
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("reflection", reflection_node)

    # 3. 设置入口点
    # (Agent 总是从 "planner" 节点开始)
    workflow.set_entry_point("planner")

    # 4. 连接“节点” (添加边)

    # (规划师 -> 编码器)
    workflow.add_edge("planner", "code_generator")

    # (编码器 -> 执行器)
    workflow.add_edge("code_generator", "code_executor")

    # (执行器 -> 反思)
    workflow.add_edge("code_executor", "reflection")

    # 5. (关键) 添加“条件边” (The Loop)
    # (在 "reflection" 节点之后，我们需要决定去哪里)

    def route_after_reflection(state: AgentState):
        """
        读取 "reflection" 节点设置的 "next_node" 状态，
        并返回要跳转到的节点的名称。
        """
        # (我们假设 reflection_node 会返回一个 'next_node' 键)
        # (这个键是在 reflection.py 中设置的)
        next_node = state.get("next_node", "continue")  # 默认为 "continue"
        print(f"--- [Graph Router] 路由决策: {next_node} ---")
        return next_node

    workflow.add_conditional_edges(
        "reflection",  # 起始节点
        route_after_reflection,  # 调用此函数来做决策
        {
            "continue": "code_generator",  # 如果返回 "continue", 跳回编码器
            "complete": END,  # 如果返回 "complete", 结束图
        },
    )

    # 6. 编译图
    print("--- [Graph] 编译完成。 ---")
    return workflow.compile()


# (关键!) 创建一个我们可以从其他地方导入的已编译的 app
app = create_agent_graph()
