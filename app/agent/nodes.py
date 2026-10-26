"""Define the steps that read and update the agent state."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

from app.agent.executor import execute_tool_call
from app.agent.model import create_model
from app.agent.state import AgentState
from app.agent.tools import ALL_AGENT_TOOLS, WRITE_TOOLS

# Call the model and return the updated messages and call count.
def model_node(state: AgentState) -> AgentState:
    if state["model_calls"] >= state["max_rounds"]:
        raise RuntimeError("the model call limit was reached")

    # Call model and collect response
    model = create_model().bind_tools(ALL_AGENT_TOOLS)
    response = model.invoke(state["messages"])
    if response.invalid_tool_calls:
        raise RuntimeError("the model returned an invalid tool call")

    # Add the model response, add call count
    return {
        **state,
        "messages": state["messages"] + [response],
        "model_calls": state["model_calls"] + 1,
    }


# Ask for approval before write calls.
def approval_node(state: AgentState) -> dict:

    # Check if call contains write tool
    calls = [call for call in state["messages"][-1].tool_calls
             if call["name"] in {item.name for item in WRITE_TOOLS}]
    
    # If contained, interrupt and wait for approval
    if calls: 
        write_approved = interrupt({"type": "write_approval", "tool_calls": calls})
    else: 
        write_approved = False

    # Return the write approval status.
    return {"write_approved": write_approved is True}


# Wait for a new message or an exit request.
def wait_node(state: AgentState) -> dict:
    question = interrupt({
        "type": "user_input", "answer": state["messages"][-1].content,
    }).strip()

    # Exit 
    if question.lower() == "q":
        return {"exit_requested": True}

    # Keep the messages and reset the model call count for the new turn.
    return {
        "messages": state["messages"] + [HumanMessage(content=question)],
        "model_calls": 0,
        "write_approved": False,
        "exit_requested": False,
    }


# Run the tools request
def tool_node(state: AgentState) -> AgentState:

    messages = state["messages"]
    if not messages or not isinstance(messages[-1], AIMessage):
        raise ValueError("the last message must be a model response")
    if state["model_calls"] >= state["max_rounds"]:
        raise RuntimeError("the model call limit was reached")

    response = messages[-1]
    tool_results = []

    for call in response.tool_calls:
        # If write tool, check write approval
        if call["name"] in {item.name for item in WRITE_TOOLS}:
            if state.get("write_approved") is True:
                result = execute_tool_call(call, approved=True)
            else: result = ToolMessage(
                content="User declined this operation. Do not retry it.", 
                tool_call_id=call["id"]
            )
        else:
            result = execute_tool_call(call)

        tool_results.append(result)

    # Add the tool results
    # Reset write approval for the next round.
    return {
        **state,
        "messages": messages + tool_results,
        "write_approved": False,
    }
