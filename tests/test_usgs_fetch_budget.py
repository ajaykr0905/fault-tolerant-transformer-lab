import io
import urllib.error

import pytest

from fttl.usgs import (
    MAX_RESPONSE_BYTES,
    MAX_SNAPSHOT_BYTES,
    USGS_FEEDS,
    USGSValidationError,
    fetch_feed,
)

FEED_URL = USGS_FEEDS["all-day"]


class _IntegerSubclass(int):
    pass


class _TrackedResponse(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


@pytest.mark.parametrize(
    "budget",
    [
        True,
        False,
        0,
        -1,
        1.5,
        "1",
        None,
        pytest.param(_IntegerSubclass(1), id="integer-subclass"),
        MAX_RESPONSE_BYTES + 1,
        2**63,
        pytest.param(10**500, id="unbounded-integer"),
    ],
)
def test_invalid_budget_fails_before_opener_or_backoff(budget):
    calls = []

    def opener(*_args):
        calls.append("open")
        return io.BytesIO(b"bounded fixture")

    with pytest.raises(ValueError, match="max_response_bytes"):
        fetch_feed(
            FEED_URL,
            opener=opener,
            sleeper=lambda _delay: calls.append("sleep"),
            max_response_bytes=budget,
        )
    assert calls == []


@pytest.mark.parametrize(
    "budget, content", [(1, b"x"), (4, b"four"), (MAX_RESPONSE_BYTES, b"small public fixture")]
)
def test_valid_budget_reads_only_one_sentinel_byte_and_closes_response(budget, content):
    response = _TrackedResponse(content)
    sleeps = []
    assert MAX_RESPONSE_BYTES == MAX_SNAPSHOT_BYTES == 64 * 1024 * 1024
    assert (
        fetch_feed(
            FEED_URL,
            opener=lambda *_args: response,
            sleeper=sleeps.append,
            max_response_bytes=budget,
        )
        == content
    )
    assert response.read_sizes == [budget + 1]
    assert response.closed and sleeps == []


def test_oversized_response_is_not_retried_and_response_is_closed():
    response = _TrackedResponse(b"one byte beyond the boundary")
    calls = []
    sleeps = []

    def opener(*_args):
        calls.append("open")
        return response

    with pytest.raises(USGSValidationError, match="byte limit"):
        fetch_feed(FEED_URL, opener=opener, sleeper=sleeps.append, max_response_bytes=4, retries=5)
    assert calls == ["open"] and sleeps == []
    assert response.read_sizes == [5] and response.closed


def test_transient_read_error_keeps_the_same_budget_and_retry_schedule():
    class InterruptedResponse(_TrackedResponse):
        def read(self, size=-1):
            self.read_sizes.append(size)
            raise OSError("independent transient read fixture")

    interrupted = InterruptedResponse(b"not returned")
    successful = _TrackedResponse(b"four")
    attempts = []
    sleeps = []

    def opener(request, timeout):
        attempts.append((request.full_url, timeout))
        return interrupted if len(attempts) == 1 else successful

    assert (
        fetch_feed(
            FEED_URL,
            opener=opener,
            sleeper=sleeps.append,
            max_response_bytes=4,
            retries=3,
        )
        == b"four"
    )
    assert attempts == [(FEED_URL, 15.0), (FEED_URL, 15.0)] and sleeps == [1.0]
    assert interrupted.read_sizes == successful.read_sizes == [5]
    assert interrupted.closed and successful.closed


def test_exhausted_transport_errors_keep_the_bounded_retry_policy():
    failure = urllib.error.URLError("independent offline transport fixture")
    calls = []
    sleeps = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        raise failure

    with pytest.raises(RuntimeError, match="after 3 attempts") as result:
        fetch_feed(FEED_URL, opener=opener, sleeper=sleeps.append, max_response_bytes=1, retries=3)
    assert result.value.__cause__ is failure
    assert calls == [(FEED_URL, 15.0)] * 3 and sleeps == [1.0, 2.0]
