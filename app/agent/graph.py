"""Connect the model and tool nodes."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import approval_node, model_node, tool_node
from app.agent.state import AgentState

# Select tool node or end graph
def route_after_model(state: AgentState) -> str:
    response = state["messages"][-1]
    if response.tool_calls:
        return "tools"
    else:
        return "end"


# Build graph with nodes and edges
def build_graph(checkpointer=None) -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("model", model_node)
    graph.add_node("tool", tool_node)
    graph.add_node("approval", approval_node)

    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", 
        route_after_model,
        {"tools": "approval", "end": END}
    )
    graph.add_edge("approval", "tool")
    graph.add_edge("tool", "model")

    return graph.compile(checkpointer=checkpointer)
