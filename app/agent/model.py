"""Connect to the DeepSeek service."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


def create_model() -> ChatDeepSeek:
    """Read the API key and create the model client."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("Set DEEPSEEK_API_KEY in the project .env file")

    return ChatDeepSeek(
        model="deepseek-flash",
        temperature=0,
        max_tokens=512,
        timeout=30,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
