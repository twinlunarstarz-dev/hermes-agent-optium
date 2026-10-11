"""A rate-limit on a fallback model must not bench the primary when only the provider matches.

Regression: free-tier 429s on openrouter ``:free`` fallbacks armed a cooldown on the paid
openrouter primary until the free-quota reset (~8h), pinning every turn to the fallback.
"""

import time
from types import SimpleNamespace

from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import _arm_rate_limit_cooldown

PRIMARY_MODEL = "anthropic/claude-haiku-5.5"
PRIMARY_PROVIDER = "openrouter"


def _agent(model, provider, fallback_activated):
    return SimpleNamespace(
        model=model,
        provider=provider,
        _primary_runtime={"model": PRIMARY_MODEL, "provider": PRIMARY_PROVIDER},
        _fallback_activated=fallback_activated,
        _rate_limit_backoff_count=0,
    )


def test_free_fallback_429_under_same_provider_does_not_bench_primary():
    agent = _agent("nvidia/nemotron-3-ultra-550b-a55b:free", PRIMARY_PROVIDER, fallback_activated=True)

    assert _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit, reset_at=None) is None
    assert not hasattr(agent, "_rate_limited_until")


def test_primary_429_still_arms_cooldown():
    agent = _agent(PRIMARY_MODEL, PRIMARY_PROVIDER, fallback_activated=False)

    seconds = _arm_rate_limit_cooldown(agent, FailoverReason.rate_limit, reset_at=None)

    assert seconds == 60
    assert agent._rate_limited_until > time.monotonic()


def test_non_rate_limit_failure_never_arms_cooldown():
    agent = _agent(PRIMARY_MODEL, PRIMARY_PROVIDER, fallback_activated=False)

    assert _arm_rate_limit_cooldown(agent, FailoverReason.server_error, reset_at=None) is None
    assert not hasattr(agent, "_rate_limited_until")
