"""Run a requested tool and return its result as a message."""

from langchain_core.messages import ToolCall, ToolMessage

from app.agent.tools import ALL_AGENT_TOOLS, WRITE_TOOLS


def execute_tool_call(call: ToolCall, *, approved: bool = False) -> ToolMessage:

    if not isinstance(call, dict) or call.get("type") != "tool_call":
        raise ValueError("call must be a tool_call dictionary")
    if not isinstance(call.get("id"), str) or not call["id"].strip():
        raise ValueError("tool call id must be a nonempty string")
    if not isinstance(call.get("name"), str) or not isinstance(call.get("args"), dict):
        raise ValueError("tool call must contain a name string and an args dictionary")

    # Select a tool from the allowed list
    tools_by_name = {item.name: item for item in ALL_AGENT_TOOLS}
    selected_tool = tools_by_name.get(call["name"])
    if selected_tool is None:
        raise ValueError("the requested tool is not available")
    if selected_tool.name in {item.name for item in WRITE_TOOLS} and approved is not True:
        raise ValueError("write tools require user approval")

    # Check the arguments and run tool functions
    # Return its result
    return selected_tool.invoke(call)
