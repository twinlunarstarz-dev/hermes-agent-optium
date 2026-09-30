"""Unit test: a CDP supervisor whose endpoint is permanently dead must give up
after ``MAX_POST_ATTACH_RECONNECT_FAILURES`` consecutive connect failures instead of
WARNING-spamming forever (a dead local agent-browser daemon / expired cloud
session never comes back). A healthy endpoint that briefly restarts reconnects
long before the cap because each successful attach resets the failure count."""

from __future__ import annotations

import asyncio

from tools.browser_supervisor import CDPSupervisor, MAX_POST_ATTACH_RECONNECT_FAILURES


def test_run_gives_up_after_max_connect_failures(monkeypatch):
    sup = CDPSupervisor(task_id="t-giveup", cdp_url="ws://127.0.0.1:1/dead")
    # Pretend the first attach already succeeded so _fail_start() returns False
    # and the reconnect loop (not the fatal-start path) handles the failures.
    sup._ready_event.set()

    attempts = {"n": 0}

    async def _refuse(*_a, **_k):
        attempts["n"] += 1
        raise ConnectionRefusedError("connect refused")

    _real_sleep = asyncio.sleep

    async def _sleep_no_wait(_delay, *a, **k):
        # Burn backoff delays instantly but keep cooperative yields (sleep(0)).
        return await _real_sleep(0)

    import websockets
    monkeypatch.setattr(websockets, "connect", _refuse)
    monkeypatch.setattr(asyncio, "sleep", _sleep_no_wait)

    asyncio.run(sup._run())

    assert attempts["n"] == MAX_POST_ATTACH_RECONNECT_FAILURES
    assert sup._start_error is None  # gave up, not crashed


def test_run_resets_after_successful_attach(monkeypatch):
    """The give-up cap triggers only on CONSECUTIVE failures: a supervisor that
    flaps (fail, connect, drop, fail, connect...) must never give up."""
    sup = CDPSupervisor(task_id="t-recover", cdp_url="ws://127.0.0.1:1/flap")
    sup._ready_event.set()

    state = {"attempts": 0, "connected_once": False}

    class _FakeWS:
        async def close(self):
            pass

    async def _connect(*_a, **_k):
        state["attempts"] += 1
        # Fail once, then let the next connect succeed (simulating a restart).
        if state["attempts"] == 1:
            raise ConnectionRefusedError("refused")
        return _FakeWS()

    _real_sleep = asyncio.sleep

    async def _sleep_no_wait(_delay, *a, **k):
        return await _real_sleep(0)

    async def _attach(self_):
        if state["connected_once"]:
            sup._stop_requested = True
            raise ConnectionResetError("stopped")
        state["connected_once"] = True
        raise ConnectionResetError("socket dropped")

    import websockets
    monkeypatch.setattr(websockets, "connect", _connect)
    monkeypatch.setattr(asyncio, "sleep", _sleep_no_wait)
    monkeypatch.setattr(CDPSupervisor, "_attach_initial_page", _attach)

    asyncio.run(sup._run())

    # Sequence: fail -> connect+drop -> connect+drop(stop). The failure counter
    # reset on each successful attach, so the give-up cap never fired and the
    # loop ended via stop_requested, not the cap.
    assert state["attempts"] == 3
    assert sup._start_error is None
    assert sup._stop_requested  # ended by stop, not by the cap
