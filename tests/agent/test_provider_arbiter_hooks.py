"""Integration tests for the provider-arbiter seams in the turn loop.

Covers the two places core code consults agent/provider_cost_arbiter.py:

* ``agent.chat_completion_helpers._provider_preferences_for_agent`` - folds the computed
  provider order into the per-request OpenRouter ``provider`` object.
* ``agent.turn_context_compaction._engine_preflight_maintenance`` - performs exactly one
  compaction pass when a provider switch is pending.
"""

from __future__ import annotations

import pytest

from agent import provider_cost_arbiter as A
from agent import turn_context_compaction as TC
from agent.chat_completion_helpers import _provider_preferences_for_agent


class _Agent:
    def __init__(self, **kw):
        self.model = kw.get("model", "deepseek/deepseek-v4.1-flash")
        self.provider = kw.get("provider", "Decart")
        self.providers_allowed = kw.get("allowed", [])
        self.providers_ignored = kw.get("ignored", [])
        self.providers_order = kw.get("order", [])
        self.provider_sort = None
        self.provider_require_parameters = True
        self.provider_data_collection = None
        self.provider_allow_fallbacks = False
        self.provider_quantizations = kw.get("quantizations", ["fp4"])


# --------------------------------------------------------------------------------------
# hook 2 - provider preferences
# --------------------------------------------------------------------------------------

def test_arbiter_override_replaces_the_configured_order(monkeypatch):
    monkeypatch.setattr(
        A, "provider_overrides_for_agent",
        lambda agent: {"order": ["sail-research", "decart"], "allow_fallbacks": True},
    )
    prefs = _provider_preferences_for_agent(_Agent(order=["decart"]))
    assert prefs["order"] == ["sail-research", "decart"]
    assert prefs["allow_fallbacks"] is True
    assert prefs["quantizations"] == ["fp4"]


def test_arbiter_override_wins_over_a_strict_pin(monkeypatch):
    """The configured pin is ``allow_fallbacks: false``; a switch must re-enable fallbacks."""
    monkeypatch.setattr(
        A, "provider_overrides_for_agent",
        lambda agent: {"order": ["relace"], "allow_fallbacks": True},
    )
    prefs = _provider_preferences_for_agent(_Agent(order=["decart"]))
    assert prefs["order"] == ["relace"]
    assert prefs["allow_fallbacks"] is True


def test_no_override_leaves_the_configured_routing_untouched(monkeypatch):
    monkeypatch.setattr(A, "provider_overrides_for_agent", lambda agent: None)
    prefs = _provider_preferences_for_agent(_Agent(order=["decart"]))
    assert prefs["order"] == ["decart"]
    assert prefs["allow_fallbacks"] is False  # the configured strict pin is preserved


def test_a_broken_arbiter_never_breaks_request_building(monkeypatch):
    def _boom(agent):
        raise RuntimeError("arbiter exploded")
    monkeypatch.setattr(A, "provider_overrides_for_agent", _boom)
    prefs = _provider_preferences_for_agent(_Agent(order=["decart"]))
    assert prefs["order"] == ["decart"]


def test_quantization_filter_survives_the_arbiter_merge(monkeypatch):
    """The fp4/fp8 request must not be widened by the override."""
    monkeypatch.setattr(
        A, "provider_overrides_for_agent",
        lambda agent: {"order": ["sail-research"]},
    )
    prefs = _provider_preferences_for_agent(_Agent(quantizations=["fp8"]))
    assert prefs["quantizations"] == ["fp8"]


# --------------------------------------------------------------------------------------
# hook 3 - compaction on switch
# --------------------------------------------------------------------------------------

class _Out:
    def __init__(self):
        self.messages = [{"role": "user", "content": "hello"}]
        self.active_system_prompt = "sys"
        self.conversation_history = []
        self.compressed = False
        self.blocked = False


class _Compressor:
    name = "lcm"
    threshold_tokens = 100000
    context_length = 200000

    def __init__(self, wants=False):
        self._wants = wants

    def should_compress_preflight(self, messages):
        return self._wants


class _LoopAgent(_Agent):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def _compress_context(self, messages, system_message, **kwargs):
        self.calls.append(kwargs)
        return [{"role": "user", "content": "compacted"}], "sys"


def test_pending_switch_triggers_exactly_one_compaction_pass():
    agent = _LoopAgent()
    A.request_compaction(agent, A.Decision(action="switch", reason="t", model=agent.model,
                                          current="Decart", chosen="Sail Research"))
    out = _Out()
    TC._engine_preflight_maintenance(agent, out, _Compressor(), 90000, "sys", "default")
    assert len(agent.calls) == 1
    assert out.compressed is True
    assert A.pending_switch_compaction(agent) is False


def test_the_compaction_request_is_not_replayed_on_the_next_turn():
    agent = _LoopAgent()
    A.request_compaction(agent, A.Decision(action="switch", reason="t", model=agent.model))
    TC._engine_preflight_maintenance(agent, _Out(), _Compressor(), 90000, "sys", "default")
    out = _Out()
    TC._engine_preflight_maintenance(agent, out, _Compressor(), 90000, "sys", "default")
    assert len(agent.calls) == 1
    assert out.compressed is False


def test_engine_driven_preflight_still_works_without_a_switch():
    agent = _LoopAgent()
    out = _Out()
    TC._engine_preflight_maintenance(agent, out, _Compressor(wants=True), 90000, "sys", "default")
    assert len(agent.calls) == 1
    assert out.compressed is True


def test_no_switch_and_no_engine_request_is_a_noop():
    agent = _LoopAgent()
    out = _Out()
    TC._engine_preflight_maintenance(agent, out, _Compressor(), 90000, "sys", "default")
    assert agent.calls == []
    assert out.compressed is False


def test_compaction_pass_is_skipped_when_the_engine_is_a_noop():
    """A no-op engine must not mark the outcome compressed (out.messages is unchanged)."""
    agent = _LoopAgent()
    agent._compress_context = lambda messages, system_message, **kw: (messages, "sys")
    A.request_compaction(agent, A.Decision(action="switch", reason="t", model=agent.model))
    out = _Out()
    TC._engine_preflight_maintenance(agent, out, _Compressor(), 90000, "sys", "default")
    assert out.compressed is False


def test_a_broken_arbiter_handshake_does_not_break_preflight(monkeypatch):
    def _boom(agent):
        raise RuntimeError("handshake exploded")
    monkeypatch.setattr(A, "pending_switch_compaction", _boom)
    agent = _LoopAgent()
    out = _Out()
    TC._engine_preflight_maintenance(agent, out, _Compressor(wants=True), 90000, "sys", "default")
    assert len(agent.calls) == 1
