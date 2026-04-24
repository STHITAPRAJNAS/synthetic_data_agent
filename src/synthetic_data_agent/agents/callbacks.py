"""Shared production callbacks for all ADK agents.

Provides four standard callbacks that are wired into every agent:
  - before_model_callback  — log request + warm up connections
  - after_model_callback   — log token usage / latency
  - before_tool_callback   — log tool start + inject timing
  - after_tool_callback    — log tool completion + latency

Using a shared module guarantees consistent observability across all specialists
without duplicating boilerplate in each agent file.
"""
from __future__ import annotations

import time
from typing import Any

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Context key used to pass call-start timestamps through CallbackContext
# ---------------------------------------------------------------------------
_MODEL_START_KEY = "_cb_model_start"
_TOOL_START_KEY = "_cb_tool_start"


# ---------------------------------------------------------------------------
# Model callbacks
# ---------------------------------------------------------------------------

async def before_model_callback(
    callback_context: Any,
    llm_request: Any,
) -> Any | None:
    """Log the outgoing LLM request and record start time.

    Returns None to let the normal LLM call proceed.
    """
    agent_name: str = getattr(callback_context, "agent_name", "unknown")
    # Stash wall-clock start in state so after_model_callback can compute latency.
    try:
        callback_context.state[_MODEL_START_KEY] = time.monotonic()
    except Exception:
        pass

    # Count messages if the request exposes them
    message_count: int = 0
    try:
        message_count = len(llm_request.contents or [])
    except Exception:
        pass

    logger.info(
        "llm_request_start",
        agent=agent_name,
        message_count=message_count,
    )
    return None  # proceed with the LLM call


async def after_model_callback(
    callback_context: Any,
    llm_response: Any,
) -> Any | None:
    """Log LLM response metadata (token usage, latency).

    Returns None to pass the response through unchanged.
    """
    agent_name: str = getattr(callback_context, "agent_name", "unknown")

    latency_ms: float = 0.0
    try:
        start = callback_context.state.get(_MODEL_START_KEY, time.monotonic())
        latency_ms = (time.monotonic() - start) * 1000
    except Exception:
        pass

    # Extract token counts if the SDK exposes them
    input_tokens: int | None = None
    output_tokens: int | None = None
    try:
        usage = llm_response.usage_metadata
        if usage:
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
    except Exception:
        pass

    logger.info(
        "llm_response_complete",
        agent=agent_name,
        latency_ms=round(latency_ms, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return None  # pass response through unchanged


# ---------------------------------------------------------------------------
# Tool callbacks
# ---------------------------------------------------------------------------

async def before_tool_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
) -> dict[str, Any] | None:
    """Log tool invocation start and record timing state.

    Returns None to let the tool execute normally.
    """
    tool_name: str = getattr(tool, "name", str(tool))
    agent_name: str = "unknown"
    try:
        agent_name = tool_context.agent_name
    except Exception:
        pass

    # Log arg keys only — never log values (may contain sensitive metadata)
    arg_keys = list(args.keys())

    logger.info(
        "tool_start",
        agent=agent_name,
        tool=tool_name,
        arg_keys=arg_keys,
    )

    # Stash timing in tool_context state
    try:
        tool_context.state[f"{_TOOL_START_KEY}_{tool_name}"] = time.monotonic()
    except Exception:
        pass

    return None  # proceed with tool execution


async def after_tool_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """Log tool completion, latency, and success/failure status.

    Returns None to pass the tool response through unchanged.
    """
    tool_name: str = getattr(tool, "name", str(tool))
    agent_name: str = "unknown"
    try:
        agent_name = tool_context.agent_name
    except Exception:
        pass

    latency_ms: float = 0.0
    try:
        start_key = f"{_TOOL_START_KEY}_{tool_name}"
        start = tool_context.state.get(start_key, time.monotonic())
        latency_ms = (time.monotonic() - start) * 1000
    except Exception:
        pass

    # Detect error responses — ADK wraps errors as dicts with an 'error' key
    is_error = isinstance(tool_response, dict) and "error" in tool_response

    logger.info(
        "tool_complete",
        agent=agent_name,
        tool=tool_name,
        latency_ms=round(latency_ms, 1),
        success=not is_error,
        error=tool_response.get("error") if is_error else None,
    )

    return None  # pass response through unchanged
