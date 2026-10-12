"""Tests for agent/provider_cost_arbiter.py (cost-aware OpenRouter provider arbitration)."""

from __future__ import annotations

import json
import sys
import types

import pytest

from agent import provider_cost_arbiter as A


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

def _endpoint(provider, slug, quant, prompt, completion, cache_read, *, uptime=100.0,
              context_length=1048576, latency=500.0, throughput=50.0, params=("tools", "reasoning")):
    return {
        "slug": slug, "provider": provider, "tag": "%s/%s" % (slug, quant), "quantization": quant,
        "prompt": prompt, "completion": completion, "cache_read": cache_read,
        "has_cache_price": cache_read != prompt, "discount": 0.0,
        "context_length": context_length, "uptime": uptime, "throughput": throughput,
        "latency": latency, "supported_parameters": list(params),
    }


#: Prices per token, mirroring the live fp4/fp8 deepseek endpoints.
FP4_SAIL = _endpoint("Sail Research", "sail-research", "fp4", 0.08e-6, 0.40e-6, 0.01e-6, latency=1226.0)
FP4_DECART = _endpoint("Decart", "decart", "fp4", 0.09e-6, 0.18e-6, 0.018e-6, uptime=98.4, latency=659.0)
FP4_OPENINF = _endpoint("OpenInference", "open-inference", "fp4", 0.0336e-6, 1.188e-6, 0.0336e-6, latency=5308.0)
FP8_MORPH = _endpoint("Morph", "morph", "fp8", 0.042e-6, 0.60e-6, 0.009e-6)
ALL_FP4 = [FP4_OPENINF, FP4_SAIL, FP4_DECART]
MIX = {"n": 645.0, "uncached_in": 7127.0, "cached_in": 88609.0, "out": 1137.0, "served": ["Decart"] * 645}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Point the arbiter's state at a throwaway home and clear its in-process memo."""
    monkeypatch.setattr(A, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(A, "_CATALOG_MEMO", {})
    monkeypatch.setattr(A, "_CATALOG_FAILED", {})
    yield tmp_path


class _Agent:
    def __init__(self, model="deepseek/deepseek-v4.1-flash", provider="Decart", **kw):
        self.model = model
        self.provider = provider
        self.providers_allowed = kw.get("allowed", [])
        self.providers_ignored = kw.get("ignored", [])
        self.provider_quantizations = kw.get("quantizations", [])


def _settings(**over):
    base = dict(A.DEFAULTS)
    base.update({"enabled": True, "min_uptime": None, "min_margin": 0.15,
                 "payback_horizon": 25.0, "cadence": 5, "cooldown": 30, "window": 200,
                 "compaction_factor": 0.4})
    base.update(over)
    return base


def _prime(mix=None, requests=30, provider="Decart", model="deepseek/deepseek-v4.1-flash"):
    """Write `requests` observations so the rolling mix and counters are warm."""
    mix = mix or MIX
    for _ in range(requests):
        A.observe(model, provider, uncached_input=int(mix["uncached_in"]),
                  cached_input=int(mix["cached_in"]), output=int(mix["out"]))
    return model


# --------------------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------------------

def test_expected_cost_is_three_term_measured_mix():
    cost = A.expected_cost(FP4_SAIL, MIX)
    expect = 7127 * 0.08e-6 + 88609 * 0.01e-6 + 1137 * 0.40e-6
    assert cost == pytest.approx(expect, rel=1e-9)
    # Sanity: the cache line dominates for this workload, which is the whole point.
    terms = {"uncached": 7127 * 0.08e-6, "cached": 88609 * 0.01e-6, "output": 1137 * 0.40e-6}
    assert terms["cached"] == max(terms.values())


def test_expected_cost_is_not_warm_plus_cold():
    """Regression: the first implementation returned warm+cold, doubling every figure."""
    sail = A.expected_cost(FP4_SAIL, MIX)
    cold_only = (7127 + 88609) * 0.08e-6 + 1137 * 0.40e-6
    assert sail < cold_only
    assert sail == pytest.approx(0.001911, abs=2e-6)


def test_switch_penalty_is_the_lost_cache_discount_on_the_compacted_prompt():
    penalty = A.switch_penalty(FP4_SAIL, MIX, compaction_factor=0.4)
    assert penalty == pytest.approx((7127 + 88609) * 0.4 * (0.08e-6 - 0.01e-6), rel=1e-9)


def test_switch_penalty_is_zero_when_cache_read_costs_the_same_as_prompt():
    no_discount = _endpoint("X", "x", "fp8", 0.05e-6, 0.5e-6, 0.05e-6)
    assert A.switch_penalty(no_discount, MIX, compaction_factor=1.0) == 0.0


# --------------------------------------------------------------------------------------
# ranking / eligibility
# --------------------------------------------------------------------------------------

def test_rank_prefers_cheapest_expected_cost_not_cheapest_prompt_price():
    ranked = A.rank_candidates([FP4_OPENINF, FP4_SAIL, FP4_DECART], MIX)
    assert [e["provider"] for e, _ in ranked] == ["Sail Research", "Decart", "OpenInference"]
    # OpenInference is cheapest on *prompt* but by far the worst overall.
    assert FP4_OPENINF["prompt"] == min(e["prompt"] for e in ALL_FP4)


def test_rank_honours_requested_quantization():
    ranked = A.rank_candidates([FP4_SAIL, FP8_MORPH], MIX, allowed_quantizations=["fp4"])
    assert [e["provider"] for e, _ in ranked] == ["Sail Research"]


def test_rank_auto_mode_spans_quantizations():
    ranked = A.rank_candidates([FP4_SAIL, FP8_MORPH], MIX)
    assert {e["quantization"] for e, _ in ranked} == {"fp4", "fp8"}


def test_rank_honours_ignore_and_only():
    assert [e["provider"] for e, _ in A.rank_candidates(ALL_FP4, MIX, ignore=["sail-research"])] == [
        "Decart", "OpenInference"]
    assert [e["provider"] for e, _ in A.rank_candidates(ALL_FP4, MIX, only=["Decart"])] == ["Decart"]
    assert A.rank_candidates(ALL_FP4, MIX, only=["nobody"]) == []


def test_rank_filters_uptime_context_and_parameters():
    flaky = _endpoint("Flaky", "flaky", "fp4", 0.01e-6, 0.01e-6, 0.01e-6, uptime=60.0)
    small = _endpoint("Small", "small", "fp4", 0.01e-6, 0.01e-6, 0.01e-6, context_length=8192)
    no_tools = _endpoint("NoTools", "notools", "fp4", 0.01e-6, 0.01e-6, 0.01e-6, params=())
    ranked = A.rank_candidates([flaky, small, no_tools, FP4_SAIL], MIX, min_uptime=95.0, min_context=200000)
    assert [e["provider"] for e, _ in ranked] == ["Sail Research"]


# --------------------------------------------------------------------------------------
# telemetry store
# --------------------------------------------------------------------------------------

def test_observe_builds_a_rolling_mix_and_caps_the_window():
    for i in range(20):
        A.observe("m", "P", uncached_input=100, cached_input=900, output=10, window=10)
    mix = A.rolling_mix("m", window=10)
    assert mix["n"] == 10.0
    assert mix["uncached_in"] == 100.0
    assert mix["cached_in"] == 900.0
    assert mix["out"] == 10.0
    assert mix["served"] == ["P"] * 10


def test_rolling_mix_is_none_before_any_observation():
    assert A.rolling_mix("never-seen") is None


def test_observe_survives_a_corrupt_state_file():
    (A._STORE_PATH()).parent.mkdir(parents=True, exist_ok=True)
    A._STORE_PATH().write_text("{not json", encoding="utf-8")
    A.observe("m", "P", uncached_input=1, cached_input=2, output=3)
    assert A.rolling_mix("m")["n"] == 1.0


def test_observe_response_usage_bridges_the_canonical_usage_dict():
    agent = _Agent()
    A.observe_response_usage(agent, {"prompt_tokens": 1000, "cache_read_tokens": 900, "output_tokens": 50})
    mix = A.rolling_mix(agent.model)
    assert (mix["uncached_in"], mix["cached_in"], mix["out"]) == (100.0, 900.0, 50.0)


# --------------------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------------------

def test_decide_returns_none_when_disabled(monkeypatch):
    _prime()
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: dict(A.DEFAULTS))
    assert A.decide(_Agent(), force=True) is None


def test_decide_is_none_before_the_cadence_elapses(monkeypatch):
    _prime(requests=3)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings(cadence=5))
    assert A.decide(_Agent()) is None


def test_decide_switches_when_the_payback_fits_the_horizon(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings(quantizations=None))
    agent = _Agent(quantizations=["fp4"])
    decision = A.decide(agent, force=True)
    assert decision is not None and decision.action == "switch"
    assert decision.chosen == "Sail Research"
    assert decision.payback_requests < 25.0
    assert decision.compaction_requested is True
    override = A.override_from_decision(decision)
    assert override["allow_fallbacks"] is True
    assert override["order"][0] == "sail-research"


def test_decide_holds_when_the_payback_exceeds_the_horizon(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings(payback_horizon=1.0))
    decision = A.decide(_Agent(quantizations=["fp4"]), force=True)
    assert decision.action == "hold"
    assert "payback" in decision.reason
    assert A.override_from_decision(decision) is None


def test_decide_holds_when_the_margin_is_below_threshold(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings(min_margin=0.99))
    decision = A.decide(_Agent(quantizations=["fp4"]), force=True)
    assert decision.action == "hold"
    assert "margin" in decision.reason


def test_decide_holds_when_already_on_the_cheapest(monkeypatch):
    _prime(requests=10, provider="Sail Research")
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())
    decision = A.decide(_Agent(provider="Sail Research", quantizations=["fp4"]), force=True)
    assert decision.action == "hold"
    assert decision.reason == "already cheapest"


def test_decide_allows_the_first_correction_without_a_prior_switch(monkeypatch):
    """No switch has ever happened, so the cooldown must not delay the initial correction."""
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings(cooldown=1000))
    decision = A.decide(_Agent(quantizations=["fp4"]))
    assert decision is not None and decision.action == "switch"


def test_decide_respects_the_cooldown_after_a_switch(monkeypatch):
    """Once we have switched, a long cooldown must block a second one even for a cheaper peer."""
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings(cooldown=1000))
    agent = _Agent(quantizations=["fp4"])
    assert A.decide(agent, force=True).action == "switch"
    assert A.decide(agent) is None  # switches == 1, since_switch == 0 < 1000


def test_decide_is_none_without_enough_data(monkeypatch):
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())
    assert A.decide(_Agent(), force=True) is None


def test_decide_is_none_when_no_endpoint_is_eligible(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())
    # fp8 requested, but every fixture endpoint is fp4 -> nothing eligible.
    assert A.decide(_Agent(quantizations=["fp8"]), force=True) is None


def test_decide_is_none_when_the_catalog_is_unavailable(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: None)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())
    assert A.decide(_Agent(quantizations=["fp4"]), force=True) is None


def test_decide_survives_a_broken_catalog(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())

    def _boom(*a, **k):
        raise RuntimeError("catalog exploded")
    monkeypatch.setattr(A, "fetch_endpoints", _boom)
    assert A.decide(_Agent(quantizations=["fp4"]), force=True) is None


def test_switch_marks_when_the_current_provider_is_ineligible(monkeypatch):
    _prime(requests=10, provider="Flaky")
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())
    decision = A.decide(_Agent(provider="Flaky", quantizations=["fp4"]), force=True)
    assert decision.action == "switch"
    assert decision.reason == "current provider not eligible"


def test_decision_order_keeps_a_failover_tail(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings(keep_fallbacks=2))
    decision = A.decide(_Agent(quantizations=["fp4"]), force=True)
    assert decision.order == ["sail-research", "decart", "open-inference"]


def test_decision_is_persisted_and_audited(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())
    A.decide(_Agent(quantizations=["fp4"]), force=True)
    store = json.loads(A._STORE_PATH().read_text(encoding="utf-8"))
    assert store["decisions"]["deepseek/deepseek-v4.1-flash"]["chosen"] == "Sail Research"
    lines = [json.loads(l) for l in A._AUDIT_PATH().read_text(encoding="utf-8").splitlines()]
    assert any(l["event"] == "decision" and l["action"] == "switch" for l in lines)


# --------------------------------------------------------------------------------------
# compaction handshake
# --------------------------------------------------------------------------------------

def test_compaction_request_is_consumed_exactly_once():
    agent = _Agent()
    from agent.provider_cost_arbiter import Decision
    A.request_compaction(agent, Decision(action="switch", reason="t", model="m", current="A", chosen="B"))
    assert A.pending_switch_compaction(agent) is True
    assert A.consume_compaction_request(agent)["to"] == "B"
    assert A.pending_switch_compaction(agent) is False
    assert A.consume_compaction_request(agent) is None


def test_provider_overrides_requests_compaction_on_switch(monkeypatch):
    _prime(requests=10)
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())
    agent = _Agent(quantizations=["fp4"])
    override = A.provider_overrides_for_agent(agent)
    assert override and override["order"][0] == "sail-research"
    assert A.pending_switch_compaction(agent) is True


def test_provider_overrides_is_none_on_hold(monkeypatch):
    _prime(requests=10, provider="Sail Research")
    monkeypatch.setattr(A, "fetch_endpoints", lambda *a, **k: ALL_FP4)
    monkeypatch.setattr(A, "load_settings", lambda cfg=None: _settings())
    assert A.provider_overrides_for_agent(_Agent(provider="Sail Research", quantizations=["fp4"])) is None
    assert A.pending_switch_compaction(_Agent()) is False


# --------------------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------------------

def test_normalize_endpoint_derives_the_order_slug_from_the_tag():
    raw = {"provider_name": "Sail Research", "tag": "sail-research/fp4", "quantization": "fp4",
           "pricing": {"prompt": "0.00000008", "completion": "0.0000004", "input_cache_read": "0.00000001"},
           "context_length": 1048576, "uptime_last_1d": 100,
           "throughput_last_30m": {"p50": 42}, "latency_last_30m": {"p50": 1226},
           "supported_parameters": ["tools", "reasoning"]}
    endpoint = A._normalize_endpoint(raw)
    assert endpoint["slug"] == "sail-research"
    assert endpoint["prompt"] == 0.00000008
    assert endpoint["context_length"] == 1048576
    assert endpoint["throughput"] == 42


def test_normalize_endpoint_falls_back_to_prompt_price_when_cache_read_is_absent():
    raw = {"provider_name": "Relace", "tag": "relace",
           "pricing": {"prompt": "0.00000005", "completion": "0.0000006"},
           "supported_parameters": ["tools"]}
    endpoint = A._normalize_endpoint(raw)
    assert endpoint["has_cache_price"] is False
    assert endpoint["cache_read"] == endpoint["prompt"]


def test_normalize_endpoint_rejects_entries_without_prices():
    assert A._normalize_endpoint({"provider_name": "X", "tag": "x", "pricing": {}}) is None


def test_fetch_endpoints_uses_the_disk_cache_without_network(tmp_path, monkeypatch):
    cached = {"models": {"m": {"fetched_at": A.time.time(), "endpoints": ALL_FP4}}}
    A._ENDPOINT_CACHE_PATH().parent.mkdir(parents=True, exist_ok=True)
    A._ENDPOINT_CACHE_PATH().write_text(json.dumps(cached), encoding="utf-8")
    calls = []

    class _Requests(types.SimpleNamespace):
        @staticmethod
        def get(*a, **k):
            calls.append(a)
            raise AssertionError("network must not be touched when the cache is warm")

    monkeypatch.setitem(sys.modules, "requests", _Requests())
    assert A.fetch_endpoints("m", ttl=3600) == ALL_FP4
    assert calls == []


def test_fetch_endpoints_returns_none_without_an_api_key(monkeypatch):
    monkeypatch.setattr(A, "_openrouter_key", lambda: None)
    assert A.fetch_endpoints("m", ttl=0) is None


def test_load_settings_merges_the_autoselect_block(monkeypatch):
    cfg = {"provider_routing": {"autoselect": {"enabled": True, "cadence": 7, "bogus": 1}}}
    settings = A.load_settings(cfg)
    assert settings["enabled"] is True
    assert settings["cadence"] == 7
    assert settings["bogus"] == 1
    assert settings["window"] == A.DEFAULTS["window"]


def test_load_settings_tolerates_a_broken_config(monkeypatch):
    class _Boom:
        def get(self, *a, **k):
            raise RuntimeError("config exploded")
    monkeypatch.setattr(A, "_per_model_settings", lambda m, c: {})
    assert A.load_settings(_Boom())["enabled"] is False


def test_fetch_endpoints_negative_cache_avoids_repeat_network_attempts(monkeypatch):
    """A model with no catalog (local/custom) must not pay a fetch every cadence."""
    monkeypatch.setattr(A, "_openrouter_key", lambda: "sk-test")
    calls = []

    class _Requests:
        @staticmethod
        def get(*a, **k):
            calls.append(a)
            raise RuntimeError("connection refused")

    monkeypatch.setitem(sys.modules, "requests", _Requests)
    assert A.fetch_endpoints("Qwen3.8-27B-Q4-Layer", ttl=3600) is None
    assert A.fetch_endpoints("Qwen3.8-27B-Q4-Layer", ttl=3600) is None
    assert A.fetch_endpoints("Qwen3.8-27B-Q4-Layer", ttl=3600) is None
    assert len(calls) == 1


def test_fetch_endpoints_negative_cache_expires(monkeypatch):
    monkeypatch.setattr(A, "_openrouter_key", lambda: "sk-test")
    calls = []

    class _Requests:
        @staticmethod
        def get(*a, **k):
            calls.append(a)
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "requests", _Requests)
    A.fetch_endpoints("m", ttl=1)
    A._CATALOG_FAILED["m"] = A.time.time() - 10  # pretend the TTL lapsed
    A.fetch_endpoints("m", ttl=1)
    assert len(calls) == 2
