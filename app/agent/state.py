"""Define the data that passes between agent nodes."""

from typing import NotRequired, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Store the user, model, and tool messages.
    messages: list[BaseMessage]

    # Count the completed model calls.
    model_calls: int

    # Set the maximum number of model calls.
    max_rounds: int

    # Record whether the user requested to exit.
    exit_requested: bool

    # Store approval for the current write calls.
    write_approved: NotRequired[bool]
