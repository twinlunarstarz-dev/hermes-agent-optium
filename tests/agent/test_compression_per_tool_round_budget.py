"""Regression tests: the compression attempt budget is scoped to one tool round.

A single turn-wide budget (``compression.max_attempts``, default 3) was spent by the
first pre-API passes and never refreshed, so every later tool call grew the request
unchecked. Observed in production: a 150k threshold with the prompt climbing past 300k.
A completed tool round now starts a fresh budget and clears the previous pass's
no-progress latch and pressure baseline.
"""

from __future__ import annotations

import contextlib
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_response(i: int):
    msg = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[
            SimpleNamespace(
                id=f"call_{i}",
                type="function",
                function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
            )
        ],
    )
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _stop_response():
    msg = SimpleNamespace(content="done", reasoning_content=None, reasoning=None, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _compressor(threshold: int) -> MagicMock:
    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = threshold
    compressor.context_length = 200_000
    compressor.last_prompt_tokens = 150_000
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.return_value = True
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    return compressor


@pytest.fixture()
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=12,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a._disable_streaming = True
    a.tool_delay = 0
    a.save_trajectories = False
    a.compression_enabled = True
    a.context_compressor = _compressor(threshold=10_000)
    return a


def _run_turn(agent, *, n_tool_rounds: int, estimate: int | None = None):
    """Drive one turn of ``n_tool_rounds`` tool calls; return (result, compactions per round).

    Round 0 is turn start (before the first request); round k is the compaction owned by
    the k-th tool round or the pre-API pass before the next request.
    """
    responses = [_tool_response(i) for i in range(n_tool_rounds)] + [_stop_response()]
    create = agent.client.chat.completions.create
    create.side_effect = responses
    compress_rounds: list[int] = []

    def _fake_compress(messages, system_message, **_kwargs):
        compress_rounds.append(create.call_count)
        return messages, "compressed prompt"

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_compress_context", side_effect=_fake_compress))
        stack.enter_context(patch.object(agent, "_persist_session"))
        stack.enter_context(patch.object(agent, "_save_trajectory"))
        stack.enter_context(patch.object(agent, "_cleanup_task_resources"))
        stack.enter_context(
            patch(
                "model_tools.handle_function_call",
                lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
            )
        )
        if estimate is not None:
            stack.enter_context(
                patch("agent.turn_context.estimate_request_tokens_rough", return_value=estimate)
            )
            stack.enter_context(
                patch("agent.model_metadata.estimate_request_tokens_rough", return_value=estimate)
            )
        result = agent.run_conversation("do a lot of tool work")
    return result, Counter(compress_rounds)


def test_no_progress_latch_does_not_block_next_tool_round(agent):
    """A pass that made no progress latches retries for its own round only.

    A constant over-threshold estimate makes each pass ineffective, so the pre-API
    no-progress latch engages. Every tool round must still start with a fresh pass;
    otherwise the latch from one round would carry into the next.
    """
    agent.context_compressor.threshold_tokens = 500

    result, per_round = _run_turn(agent, n_tool_rounds=6, estimate=1_000)

    assert result["completed"] is True
    cap = agent.max_compression_attempts
    assert all(1 <= per_round[rnd] <= cap for rnd in range(1, 7)), (
        "every pressured tool round must compact at least once, and no round may exceed "
        f"the cap ({cap}): {dict(per_round)}"
    )
