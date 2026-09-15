"""In-memory token bucket.

The server is single-instance and not publicly deployed, so the bucket lives
in process memory: no Redis, no persistence, and no coordination between
replicas. What it does buy is a bound on how fast a misbehaving agent loop can
hammer Qdrant and the embedding model.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class RateLimitError(RuntimeError):
    """Raised when a call arrives with no token left to spend."""

    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__(
            f"rate limit exceeded, retry in {retry_after_seconds:.1f}s"
        )
        self.retry_after_seconds = retry_after_seconds


class TokenBucket:
    """Allows `capacity` calls in a burst, then `refill_per_second` sustained."""

    def __init__(
        self,
        *,
        capacity: int,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = float(capacity)
        self._refill_per_second = refill_per_second
        self._clock = clock
        self._tokens = float(capacity)
        self._last_refill = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last_refill
        self._last_refill = now
        # Capping at capacity is what makes an idle period grant a burst of at
        # most `capacity`, instead of an allowance that grows without bound.
        self._tokens = min(
            self._capacity, self._tokens + elapsed * self._refill_per_second
        )

    def acquire(self) -> None:
        """Spend one token, or raise `RateLimitError`."""
        self._refill()
        if self._tokens < 1.0:
            missing = 1.0 - self._tokens
            raise RateLimitError(retry_after_seconds=missing / self._refill_per_second)
        self._tokens -= 1.0
