"""Per-minute rate limits are waited out; daily quotas fail fast.

The error texts are real provider responses (trimmed): waiting 65 s six
times on a daily quota only delayed the inevitable failure by six minutes.
"""

import pytest

import agent_core
from ratelimit import is_daily_quota, is_rate_limited, should_retry

GEMINI_DAILY = (
    "Error calling model 'gemini-3.6-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan "
    "and billing details. Quota exceeded for metric: generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests, limit: 20', 'details': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}, {'retryDelay': '41s'}]}}"
)
GEMINI_MINUTE = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}, {'retryDelay': '12s'}]}}"
)
GROQ_DAILY = (
    "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` service tier "
    "`on_demand` on tokens per day (TPD): Limit 200000, Used 199020, Requested 4210."
)
GROQ_MINUTE = (
    "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` service tier "
    "`on_demand` on tokens per minute (TPM): Limit 8000, Used 6100, Requested 4210. "
    "Please try again in 17.3s."
)


@pytest.mark.parametrize(
    "text, daily",
    [(GEMINI_DAILY, True), (GROQ_DAILY, True), (GEMINI_MINUTE, False), (GROQ_MINUTE, False)],
)
def test_classifies_daily_vs_per_minute(text, daily):
    exc = RuntimeError(text)
    assert is_rate_limited(exc)
    assert is_daily_quota(exc) is daily
    assert should_retry(exc) is (not daily)


def test_unrelated_errors_are_not_retried():
    exc = ValueError("tool call arguments were not valid JSON")
    assert not should_retry(exc) and not is_daily_quota(exc)


async def _run(monkeypatch, error_text):
    calls, sleeps = [], []

    async def handler(request):
        calls.append(request)
        raise RuntimeError(error_text)

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(agent_core.asyncio, "sleep", fake_sleep)
    with pytest.raises(RuntimeError):
        await agent_core.retry_on_rate_limit.awrap_model_call("req", handler)
    return calls, sleeps


async def test_daily_quota_fails_on_first_attempt(monkeypatch):
    calls, sleeps = await _run(monkeypatch, GEMINI_DAILY)
    assert len(calls) == 1 and sleeps == []


async def test_per_minute_limit_is_retried_with_the_provider_hint(monkeypatch):
    calls, sleeps = await _run(monkeypatch, GEMINI_MINUTE)
    assert len(calls) == 6  # gave up after max_attempts
    assert sleeps[0] == pytest.approx(13.0)  # retryDelay 12s + 1s cushion
