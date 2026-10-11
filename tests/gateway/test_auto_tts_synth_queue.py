"""Final-reply TTS must not gate the agent turn.

Regression test for the inline `await self._synthesize_auto_tts(...)` in
BasePlatformAdapter._process_message_background, which held the turn for the full
model render (measured 30-140s per reply with qwen3-tts on this box).

The synthesis queue makes the handoff non-blocking while still delivering audio.
"""
import asyncio
import os

import pytest

from gateway.platforms.base import BasePlatformAdapter

SYNTH_DELAY = 3.0  # stand-in for the real 30-140s qwen3-tts render


class FakeAdapter(BasePlatformAdapter):
    name = "fake"
    platform = "discord"

    def __init__(self):
        self._played = []
        self._synth_calls = 0

    async def _synthesize_auto_tts(self, text_content):
        self._synth_calls += 1
        await asyncio.sleep(SYNTH_DELAY)          # the model render
        p = "/tmp/fake_tts_%d.mp3" % self._synth_calls
        with open(p, "wb") as f:
            f.write(b"ID3" + b"\0" * 64)
        return [p], None

    async def play_tts(self, chat_id, audio_path, **kwargs):
        self._played.append((chat_id, audio_path))
        return True

    # --- abstract methods required by BasePlatformAdapter ---
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send(self, chat_id, content, **kwargs):
        pass


@pytest.mark.asyncio
async def test_enqueue_does_not_block_the_turn():
    a = FakeAdapter()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    deferred = a._enqueue_auto_tts("chat123", "final reply text", {"notify": True})
    elapsed = loop.time() - t0

    assert deferred is True, "enqueue should have accepted the job"
    assert elapsed < 0.25, "enqueue blocked %.3fs - TTS is still synchronous" % elapsed
    assert a._synth_calls == 0, "synthesis ran inline; the turn is gated on the model"


@pytest.mark.asyncio
async def test_background_worker_delivers_and_cleans_up():
    a = FakeAdapter()
    assert a._enqueue_auto_tts("chat123", "final reply text", {"notify": True}) is True
    await a._tts_synth_task

    assert a._synth_calls == 1
    assert len(a._played) == 1, "audio never reached play_tts"
    assert a._played[0][0] == "chat123"

    # The worker owns the temp file once the turn has handed off.
    leaked = [p for _, p in a._played if os.path.exists(p)]
    assert not leaked, "worker leaked temp files: %s" % leaked


@pytest.mark.asyncio
async def test_queue_full_falls_back_to_inline():
    a = FakeAdapter()
    a._tts_synth_queue_limit = 2
    assert a._enqueue_auto_tts("c", "one", {}) is True
    assert a._enqueue_auto_tts("c", "two", {}) is True
    # Third job exceeds maxsize -> caller must fall back to inline synthesis.
    assert a._enqueue_auto_tts("c", "three", {}) is False
    await a._tts_synth_task
