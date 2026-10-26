# Start agent conversations through HTTP requests.

from fastapi import APIRouter, Body, HTTPException

from app.agent.workflow import (
    ConversationNotFoundError, ConversationNotWaitingError,
    resume_agent as resume_workflow, run_agent,
)

router = APIRouter(prefix="/agent", tags=["Agent"])


# Run until the graph waits for a message or write approval.
@router.post("/start")
def start_agent(question: str = Body(..., embed=True, min_length=1, strict=True)) -> dict:
    try:
        return run_agent(question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Receive a reply and convert workflow errors to HTTP responses.
@router.post("/{thread_id}/resume")
def resume_agent(
    thread_id: str,
    reply: str = Body(..., embed=True, min_length=1, strict=True,
                      description="During chat, enter a message or q to quit. During write approval, enter y to confirm; other text cancels."),
) -> dict:
    try:
        return resume_workflow(thread_id, reply)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConversationNotWaitingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
