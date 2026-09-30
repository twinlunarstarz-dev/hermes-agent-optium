"""Focused Discord interim-voice regression tests.

The gateway's ordinary assistant commentary callback can run without a
GatewayStreamConsumer (for example when stream setup is unavailable). That
fallback must still mark the send as interim; Discord's adapter only speaks
sends carrying ``metadata["_interim_send"]``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from plugins.platforms.discord.adapter import DiscordAdapter


def test_no_stream_consumer_interim_commentary_is_marked_for_voice(monkeypatch):
    """The no-consumer fallback keeps the user-facing interim marker."""
    calls = []

    class Runner:
        # The real StreamingConfig exposes enabled_for(plat_override); a bare
        # SimpleNamespace(enabled=...) raises AttributeError here.
        from gateway.config import StreamingConfig as _SC
        config = SimpleNamespace(streaming=_SC(enabled=False, transport="off"))

        # Upstream renamed this to _delivery_adapter_for (profile-scoped resolver).
        def _delivery_adapter_for(self, _source):
            return None

        def _build_stream_consumer_config(self, *_args, **_kwargs):
            raise RuntimeError("stream setup unavailable")

    ctx = TurnContext(
        source=SimpleNamespace(platform=SimpleNamespace(value="discord"), chat_id="chat"),
        resolve_display_setting=lambda *_args: False,
        interim_assistant_messages_enabled=True,
        # Upstream gates both stream deltas and interim messages on this flag:
        # a scheduled heartbeat turn must not emit interim commentary at all.
        scheduled_heartbeat=False,
        _run_still_current=lambda: True,
        _status_adapter=object(),
        _status_chat_id="chat",
    )
    runner = TurnRunner(Runner(), ctx)
    monkeypatch.setattr(
        runner, "_send_status_text",
        lambda text, metadata, log_message: calls.append((text, metadata, log_message)),
    )

    _, _, interim, _ = runner._setup_stream_consumer("discord")
    interim("Understood. From now on, non-trivial requests will be delegated.", already_streamed=False)

    assert len(calls) == 1
    assert calls[0][0].startswith("Understood.")
    assert calls[0][1]["_interim_send"] is True


async def _run_interim_voice_once(monkeypatch, *, mode, voice_origin=None, interim=True):
    """Run the adapter hook with a fake channel/TTS sink and return TTS calls."""
    adapter = object.__new__(DiscordAdapter)
    # Mirror the REAL discord.py channel shape: TextChannel/Thread expose
    # ``channel.guild.id`` — there is no ``guild_id`` attribute in discord.py 2.x.
    # A fake with a bare ``guild_id`` attr would mask the production resolution bug.
    adapter._resolve_channel = AsyncMock(
        return_value=SimpleNamespace(guild=SimpleNamespace(id=42)))
    adapter.is_in_voice_channel = lambda _guild_id: True
    adapter._voice_mode_getter = lambda _chat_id: mode
    adapter._voice_mixers = {}
    adapter._voice_speech_queues = {}
    adapter._voice_speech_tasks = {}
    adapter._voice_speech_queue_limit = 4
    adapter._voice_fx_cfg = {}
    adapter.play_in_voice_channel = AsyncMock(return_value=True)

    tts_calls = []
    tts_finished = asyncio.Event()
    loop = asyncio.get_running_loop()

    def fake_tts(*, text, output_path):
        tts_calls.append(text)
        Path(output_path).write_bytes(b"fake-audio")
        # text_to_speech_tool runs in asyncio.to_thread; Event.set() must be marshalled
        # back to the test loop rather than called from the worker thread.
        loop.call_soon_threadsafe(tts_finished.set)
        return json.dumps({"success": True, "file_path": output_path})

    import tools.tts_tool as tts_tool
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    metadata = {"_interim_send": interim}
    if voice_origin is not None:
        metadata["_voice_origin"] = voice_origin
    await adapter._maybe_speak_interim_in_voice("chat", "This is an interim assistant acknowledgement.", metadata)

    # The hook intentionally schedules its TTS work with asyncio.create_task.  Yield until
    # that task has reached the TTS sink for cases where speech is expected; checking tts_calls
    # before yielding used to return immediately and make this harness report a false negative.
    if interim and mode != "off" and (mode != "voice_only" or voice_origin is True):
        await asyncio.wait_for(tts_finished.wait(), timeout=2)
    else:
        # Let a no-op invocation run to completion so it cannot leak into the next case.
        await asyncio.sleep(0)

    # The task performs playback after synthesis; wait for that observable side effect too.
    for _ in range(20):
        if adapter.play_in_voice_channel.await_count:
            break
        await asyncio.sleep(0.01)
    return tts_calls, adapter.play_in_voice_channel


@pytest.mark.asyncio
async def test_discord_interim_all_mode_speaks_fire_and_forget(monkeypatch):
    tts_calls, playback = await _run_interim_voice_once(monkeypatch, mode="all")
    assert tts_calls == ["This is an interim assistant acknowledgement."]
    assert playback.await_count == 1


@pytest.mark.asyncio
async def test_discord_speech_queue_preserves_arrival_order(monkeypatch):
    """The second clip cannot begin before the first clip finishes."""
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_speech_queues = {}
    adapter._voice_speech_tasks = {}
    adapter._voice_speech_queue_limit = 4
    started = []
    first_finished = asyncio.Event()

    async def play(_guild_id, path):
        started.append(path)
        if path == "first.mp3":
            await first_finished.wait()
        return True

    adapter.play_in_voice_channel = play
    adapter.enqueue_voice_speech(42, "first.mp3")
    adapter.enqueue_voice_speech(42, "second.mp3")
    await asyncio.sleep(0)
    assert started == ["first.mp3"]

    first_finished.set()
    await asyncio.wait_for(adapter._voice_speech_queues[42].join(), timeout=1)
    assert started == ["first.mp3", "second.mp3"]
    await adapter._cancel_voice_speech_queue(42)


@pytest.mark.asyncio
async def test_discord_speech_reservations_preserve_message_arrival_when_synthesis_finishes_out_of_order():
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_speech_queues = {}
    adapter._voice_speech_tasks = {}
    adapter._voice_speech_waiters = {}
    adapter._voice_speech_queue_limit = 4
    started = []
    first_finished = asyncio.Event()
    second_ready = asyncio.Event()

    async def play(_guild_id, path):
        started.append(path)
        if path == "first.mp3":
            first_finished.set()
            await second_ready.wait()
        return True

    adapter.play_in_voice_channel = play
    first_ticket = adapter.reserve_voice_speech(42)
    second_ticket = adapter.reserve_voice_speech(42)
    adapter.enqueue_voice_speech(42, "second.mp3", ticket=second_ticket)
    await asyncio.sleep(0)
    assert started == []

    adapter.enqueue_voice_speech(42, "first.mp3", ticket=first_ticket)
    await asyncio.wait_for(first_finished.wait(), timeout=1)
    await asyncio.sleep(0)
    assert started == ["first.mp3"]
    second_ready.set()
    await asyncio.wait_for(adapter._voice_speech_queues[42].join(), timeout=1)
    assert started == ["first.mp3", "second.mp3"]
    await adapter._cancel_voice_speech_queue(42)


@pytest.mark.asyncio
async def test_discord_cancelled_reservation_does_not_stall_successor():
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_speech_queues = {}
    adapter._voice_speech_tasks = {}
    adapter._voice_speech_queue_limit = 4
    played = []

    async def play(_guild_id, path):
        played.append(path)
        return True

    adapter.play_in_voice_channel = play
    failed_ticket = adapter.reserve_voice_speech(42)
    next_ticket = adapter.reserve_voice_speech(42)
    adapter.cancel_voice_speech_reservation(42, failed_ticket)
    assert adapter.enqueue_voice_speech(42, "next.mp3", ticket=next_ticket) is True

    await asyncio.wait_for(adapter._voice_speech_queues[42].join(), timeout=1)
    assert played == ["next.mp3"]
    await adapter._cancel_voice_speech_queue(42)


@pytest.mark.asyncio
async def test_discord_speech_queue_isolates_item_failure():
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_speech_queues = {}
    adapter._voice_speech_tasks = {}
    adapter._voice_speech_queue_limit = 4
    played = []

    async def play(guild_id: int, audio_path: str):
        played.append(audio_path)
        if audio_path == "bad.mp3":
            raise RuntimeError("decode failed")
        return True

    adapter.play_in_voice_channel = play
    assert adapter.enqueue_voice_speech(42, "bad.mp3") is True
    assert adapter.enqueue_voice_speech(42, "good.mp3") is True
    await asyncio.wait_for(adapter._voice_speech_queues[42].join(), timeout=1)
    assert played == ["bad.mp3", "good.mp3"]
    await adapter._cancel_voice_speech_queue(42)


@pytest.mark.asyncio
async def test_discord_speech_queue_overflow_keeps_already_queued_speech():
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_speech_queues = {42: asyncio.Queue(maxsize=1)}
    adapter._voice_speech_tasks = {42: MagicMock(done=lambda: True)}
    adapter._voice_speech_queue_limit = 1
    adapter._voice_speech_queues[42].put_nowait("queued.mp3")

    assert adapter.enqueue_voice_speech(42, "overflow.mp3") is False
    assert adapter._voice_speech_queues[42].get_nowait() == "queued.mp3"


@pytest.mark.asyncio
async def test_discord_voice_only_requires_voice_origin(monkeypatch):
    tts_calls, playback = await _run_interim_voice_once(monkeypatch, mode="voice_only")
    assert tts_calls == []
    assert playback.await_count == 0

    tts_calls, playback = await _run_interim_voice_once(monkeypatch, mode="voice_only", voice_origin=True)
    assert tts_calls == ["This is an interim assistant acknowledgement."]
    assert playback.await_count == 1


@pytest.mark.asyncio
async def test_discord_interim_reserves_message_before_synthesis_and_publishes_one_batch(monkeypatch):
    adapter = object.__new__(DiscordAdapter)
    # Real discord.py channel shape: guild id lives on channel.guild.id.
    adapter._resolve_channel = AsyncMock(
        return_value=SimpleNamespace(guild=SimpleNamespace(id=42)))
    adapter.is_in_voice_channel = lambda _guild_id: True
    adapter._voice_mode_getter = lambda _chat_id: "all"
    adapter._voice_speech_queues = {}
    adapter._voice_speech_tasks = {}
    adapter._voice_speech_queue_limit = 4
    adapter._voice_fx_cfg = {}
    adapter.reserve_voice_speech = MagicMock(return_value=7)
    adapter.enqueue_voice_speech_batch = MagicMock(return_value=True)
    adapter.enqueue_voice_speech = MagicMock(side_effect=AssertionError("must use one message batch"))
    synthesized = asyncio.Event()
    loop = asyncio.get_running_loop()

    def fake_tts(*, text, output_path):
        adapter.reserve_voice_speech.assert_called_once_with(42)
        Path(output_path).write_bytes(b"fake-audio")
        loop.call_soon_threadsafe(synthesized.set)
        return json.dumps({"success": True, "file_path": output_path})

    import tools.tts_tool as tts_tool
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)

    await adapter._maybe_speak_interim_in_voice(
        "chat", "This is a substantive interim assistant update.", {"_interim_send": True}
    )
    await asyncio.wait_for(synthesized.wait(), timeout=1)
    for _ in range(20):
        if adapter.enqueue_voice_speech_batch.call_count:
            break
        await asyncio.sleep(0.01)

    args, kwargs = adapter.enqueue_voice_speech_batch.call_args
    assert args[0] == 42
    assert len(args[1]) == 1
    assert kwargs == {"ticket": 7}


@pytest.mark.asyncio
async def test_discord_move_cancels_queued_speech_before_reconnect():
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_locks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_clients = {42: SimpleNamespace()}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._voice_text_channels = {42: 99}
    adapter._voice_sources = {}
    adapter._client = MagicMock()
    existing = adapter._voice_clients[42]
    existing.is_connected = lambda: True
    existing.channel = SimpleNamespace(id=7)
    existing.move_to = AsyncMock()
    channel = SimpleNamespace(guild=SimpleNamespace(id=42), id=8)
    cancel = AsyncMock()
    adapter._cancel_voice_speech_queue = cancel

    assert await adapter.join_voice_channel(channel) is True
    cancel.assert_awaited_once_with(42)
    existing.move_to.assert_awaited_once_with(channel)


@pytest.mark.asyncio
async def test_discord_leave_cancels_queued_speech():
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_locks = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_clients = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {42: MagicMock()}
    adapter._voice_text_channels = {42: 99}
    adapter._voice_sources = {42: {"chat_id": "99"}}
    adapter._client = MagicMock()
    cancel = AsyncMock()
    adapter._cancel_voice_speech_queue = cancel

    await adapter.leave_voice_channel(42)
    cancel.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_discord_disabled_voice_acknowledgement_never_synthesizes(tmp_path, monkeypatch):
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_fx_cfg = {"ack_enabled": False, "ack_phrases": ["Working on it."]}
    adapter._voice_mixers = {42: MagicMock()}
    adapter._lead_silence_bytes = lambda: b""
    tts = MagicMock(side_effect=AssertionError("disabled ack must not synthesize"))
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", tts)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    assert await adapter.play_ack_in_voice(42) is False
    tts.assert_not_called()
    adapter._voice_mixers[42].play_speech.assert_not_called()


@pytest.mark.asyncio
async def test_discord_voice_off_and_finals_stay_silent(monkeypatch):
    tts_calls, playback = await _run_interim_voice_once(monkeypatch, mode="off")
    assert tts_calls == []
    assert playback.await_count == 0

    tts_calls, playback = await _run_interim_voice_once(monkeypatch, mode="all", interim=False)
    assert tts_calls == []
    assert playback.await_count == 0
