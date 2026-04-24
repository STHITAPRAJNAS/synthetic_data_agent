"""Async retry utilities with exponential backoff and jitter.

Usage
-----
# One-off: wrap a coroutine factory
result = await retry_async(lambda: db.fetch_rows(table), max_attempts=3)

# Decorator on an async method / function
@async_retry(max_attempts=3, backoff_base=1.0, retryable=(IOError, TimeoutError))
async def flaky_io_call():
    ...
"""
from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger()

_T = TypeVar("_T")

# Exceptions that are always retriable regardless of caller config
_DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (
    IOError,
    TimeoutError,
    ConnectionError,
    OSError,
)


async def retry_async(
    fn: Callable[[], Coroutine[Any, Any, _T]],
    *,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
    jitter: bool = True,
    retryable: tuple[type[BaseException], ...] = _DEFAULT_RETRYABLE,
    label: str = "operation",
) -> _T:
    """Execute *fn()* up to *max_attempts* times with exponential backoff.

    Args:
        fn: Zero-argument async callable returning a coroutine.
        max_attempts: Total attempts before re-raising the last exception.
        backoff_base: Base delay in seconds (doubled each retry).
        backoff_cap: Maximum delay in seconds.
        jitter: Add ±25% random jitter to the delay.
        retryable: Exception types that trigger a retry.
        label: Human-readable label for log messages.

    Returns:
        The return value of *fn()* on success.

    Raises:
        The last exception if all attempts are exhausted.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except retryable as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == max_attempts:
                logger.error(
                    "retry_exhausted",
                    label=label,
                    attempts=attempt,
                    error=str(exc),
                )
                raise

            raw_delay = min(backoff_base * (2 ** (attempt - 1)), backoff_cap)
            delay = raw_delay * (0.75 + random.random() * 0.5) if jitter else raw_delay
            logger.warning(
                "retry_attempt",
                label=label,
                attempt=attempt,
                next_delay_s=round(delay, 2),
                error=str(exc),
            )
            await asyncio.sleep(delay)

    # Should be unreachable — satisfy type checker
    raise RuntimeError(f"retry_async: unreachable after {max_attempts} attempts") from last_exc


def async_retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
    jitter: bool = True,
    retryable: tuple[type[BaseException], ...] = _DEFAULT_RETRYABLE,
) -> Callable[[Callable[..., Coroutine[Any, Any, _T]]], Callable[..., Coroutine[Any, Any, _T]]]:
    """Decorator that wraps an async function with retry logic.

    Example::

        @async_retry(max_attempts=4, retryable=(IOError,))
        async def fetch_data(table: str) -> pd.DataFrame:
            ...
    """
    def decorator(
        func: Callable[..., Coroutine[Any, Any, _T]],
    ) -> Callable[..., Coroutine[Any, Any, _T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            return await retry_async(
                lambda: func(*args, **kwargs),
                max_attempts=max_attempts,
                backoff_base=backoff_base,
                backoff_cap=backoff_cap,
                jitter=jitter,
                retryable=retryable,
                label=func.__qualname__,
            )
        return wrapper
    return decorator
