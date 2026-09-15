"""Token bucket guarding the tool server.

Time is injected so the refill behaviour is asserted deterministically rather
than by sleeping.
"""

from __future__ import annotations

import pytest

from mcp_server.ratelimit import RateLimitError, TokenBucket


class FakeClock:
    """Monotonic clock the test moves forward by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_allows_calls_up_to_capacity() -> None:
    bucket = TokenBucket(capacity=3, refill_per_second=1.0, clock=FakeClock())

    for _ in range(3):
        bucket.acquire()


def test_rejects_the_call_that_exceeds_capacity() -> None:
    bucket = TokenBucket(capacity=2, refill_per_second=1.0, clock=FakeClock())
    bucket.acquire()
    bucket.acquire()

    with pytest.raises(RateLimitError):
        bucket.acquire()


def test_refills_over_time() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, refill_per_second=1.0, clock=clock)
    bucket.acquire()
    bucket.acquire()

    clock.advance(1.0)

    bucket.acquire()


def test_refill_is_capped_at_capacity() -> None:
    """A long idle period must not build up an unbounded burst allowance."""
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, refill_per_second=1.0, clock=clock)

    clock.advance(3600.0)

    bucket.acquire()
    bucket.acquire()
    with pytest.raises(RateLimitError):
        bucket.acquire()


def test_refills_partially() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=4, refill_per_second=2.0, clock=clock)
    for _ in range(4):
        bucket.acquire()

    clock.advance(0.5)

    bucket.acquire()
    with pytest.raises(RateLimitError):
        bucket.acquire()


def test_error_reports_how_long_to_wait() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, refill_per_second=2.0, clock=clock)
    bucket.acquire()

    with pytest.raises(RateLimitError) as excinfo:
        bucket.acquire()

    assert excinfo.value.retry_after_seconds == pytest.approx(0.5)
