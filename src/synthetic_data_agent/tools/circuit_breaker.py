"""Simple async circuit-breaker for protecting downstream dependencies.

State machine:
  CLOSED   → normal operation; failures increment a counter.
  OPEN     → all calls rejected immediately; entered after *failure_threshold*
              consecutive failures.
  HALF_OPEN → one probe call allowed after *recovery_timeout* seconds;
              success → CLOSED, failure → OPEN.

Usage
-----
from synthetic_data_agent.tools.circuit_breaker import CircuitBreaker

_cb = CircuitBreaker(name="databricks", failure_threshold=5, recovery_timeout=60)

async def get_data():
    async with _cb:
        return await spark_query(...)
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum, auto
from typing import Any

import structlog

logger = structlog.get_logger()


class _State(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreakerOpen(Exception):
    """Raised when a call is made through an OPEN circuit breaker."""


class CircuitBreaker:
    """Async context-manager circuit breaker.

    Args:
        name: Human-readable name for log messages.
        failure_threshold: Consecutive failures before tripping OPEN.
        recovery_timeout: Seconds to wait in OPEN before moving to HALF_OPEN.
        success_threshold: Consecutive successes needed in HALF_OPEN to reset.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = _State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state.name

    @property
    def is_open(self) -> bool:
        return self._state == _State.OPEN

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CircuitBreaker":
        async with self._lock:
            if self._state == _State.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        "circuit_breaker_half_open",
                        name=self.name,
                        elapsed_s=round(elapsed, 1),
                    )
                    self._state = _State.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker '{self.name}' is OPEN "
                        f"(retry in {self.recovery_timeout - elapsed:.0f}s)"
                    )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        async with self._lock:
            if exc_type is None:
                self._on_success()
            else:
                self._on_failure()
        return False  # never suppress exceptions

    # ------------------------------------------------------------------
    # Internal state transitions
    # ------------------------------------------------------------------

    def _on_success(self) -> None:
        if self._state == _State.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                logger.info("circuit_breaker_closed", name=self.name)
                self._state = _State.CLOSED
                self._failure_count = 0
        elif self._state == _State.CLOSED:
            self._failure_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == _State.HALF_OPEN or self._failure_count >= self.failure_threshold:
            logger.warning(
                "circuit_breaker_opened",
                name=self.name,
                consecutive_failures=self._failure_count,
            )
            self._state = _State.OPEN
            self._failure_count = 0  # reset for next cycle

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Force circuit breaker back to CLOSED (for testing / ops)."""
        async with self._lock:
            self._state = _State.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info("circuit_breaker_reset", name=self.name)

    def health(self) -> dict[str, Any]:
        """Return a snapshot of circuit-breaker state for /health endpoints."""
        return {
            "name": self.name,
            "state": self._state.name,
            "failure_count": self._failure_count,
            "last_failure_age_s": (
                round(time.monotonic() - self._last_failure_time, 1)
                if self._last_failure_time
                else None
            ),
        }
