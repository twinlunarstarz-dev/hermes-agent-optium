"""Unit tests for agent.turn_facade_lease (admission + lease bracket)."""
import threading
from types import SimpleNamespace

from agent.turn_facade_lease import (
    LEASE_TTL_SECONDS,
    PREEMPT_MESSAGE,
    DurableTurnLease,
    admit_durable_turn_lease,
    preempt_live_turn_leases,
)


class _Db:
    def __init__(self, exists=True, acquired=True):
        self.exists = exists
        self.acquired = acquired
        self.events = []

    def get_session(self, session_id):
        return {"id": session_id} if self.exists else None

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.events.append(("acquire", session_id, holder))
        return self.acquired

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        self.events.append(("release", session_id, holder))


def _agent(db, **overrides):
    agent = SimpleNamespace(
        _session_db=db,
        session_id="s1",
        _persist_disabled=False,
        _interrupt_requested=False,
        _interrupt_message=None,
        _execution_thread_id=None,
        _session_turn_lease_refresh_interval=60.0,
        statuses=[],
    )
    agent._emit_status = agent.statuses.append
    agent._emit_warning = agent.statuses.append
    agent._touch_activity = lambda *a, **k: None
    agent._liveness_activity_lock = lambda: threading.Lock()
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _admit(agent, history=None):
    return admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "cli"},
        conversation_history=history,
    )


def test_fresh_row_leases_with_seed_kept_and_persist_disabled_skips(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    seed = [{"role": "user", "content": "hi"}]
    agent = _agent(_Db(exists=False))
    admission = _admit(agent, seed)
    assert admission.lease is not None and admission.early_result is None
    assert admission.conversation_history is seed
    assert getattr(agent, "_session_db_created", False) is False
    admission.lease.release()

    db = _Db()
    admission = _admit(_agent(db, _persist_disabled=True), seed)
    assert admission.lease is None and db.events == []


def test_admission_sets_holder_attrs_and_release_clears_them(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    db = _Db()
    agent = _agent(db)
    admission = _admit(agent)
    lease = admission.lease
    assert isinstance(lease, DurableTurnLease)
    assert agent._session_db_created is True
    assert agent._active_session_turn_lease_holder == lease.holder
    assert agent._active_session_turn_lease_ttl_seconds == LEASE_TTL_SECONDS
    assert lease.holder.startswith("pid=") and ":platform=cli" in lease.holder
    assert lease.watchdog is None and lease.timer_handles == []
    assert lease.is_turn_active() is False

    lease.stop_refresher()
    lease.join_threads()
    lease.clear_interrupt()
    lease.release()
    assert db.events == [("acquire", "s1", lease.holder), ("release", "s1", lease.holder)]
    assert agent._active_session_turn_lease_holder is None
    assert agent._active_session_turn_lease_ttl_seconds is None


def test_timeout_and_interrupt_early_results():
    agent = _agent(_Db(acquired=False))
    admission = _admit(agent, [{"role": "user", "content": "x"}])
    assert admission.lease is None
    assert admission.early_result["failed"] is True
    assert admission.early_result["error"] == "session_turn_lease_timeout:s1"
    # Stamped so UI descriptors show "session busy" instead of code="unknown".
    assert admission.early_result["failure_reason"] == "session_busy"
    assert admission.early_result["failure_retryable"] is True
    assert admission.early_result["messages"] == [{"role": "user", "content": "x"}]

    agent = _agent(_Db(acquired=False), _interrupt_requested=True, _interrupt_message="stop")
    agent.clear_interrupt = lambda: None
    admission = _admit(agent)
    assert admission.early_result["interrupted"] is True
    assert admission.early_result["interrupt_message"] == "stop"


def test_interrupt_turn_only_while_active():
    agent = _agent(_Db())
    calls = []
    agent.interrupt = lambda msg, **kw: calls.append(msg)
    lease = DurableTurnLease(agent, agent._session_db, "s1", "h")
    lease._interrupt_turn("lost")  # inactive: ignored
    assert calls == [] and lease.interrupt_message is None
    lease.turn_active = True
    lease._interrupt_turn("lost")
    assert calls == ["lost"] and lease.interrupt_message == "lost"
    lease.deactivate_after_liveness_abort()
    assert lease.stop.is_set() and lease.is_turn_active() is False


class _SameProcessDb:
    """Holder row owned by the first admitted turn. A second turn in the SAME process waits:
    ``on_wait`` is called, and the row only frees once the first turn has stopped renewing it."""

    def __init__(self, first_lease_getter):
        self.first_lease_getter = first_lease_getter
        self.wait_calls = 0

    def get_session(self, session_id):
        return {"id": session_id}

    def resolve_resume_session_id(self, session_id):
        return None

    def get_messages_as_conversation(self, session_id, **kwargs):
        return []

    def acquire_session_turn_lease(self, session_id, holder, on_wait=None, **kwargs):
        first = self.first_lease_getter()
        if first is None:
            return True
        for _ in range(3):
            self.wait_calls += 1
            if on_wait is not None:
                on_wait(0.0)
            if not first.is_turn_active():  # its renewal stopped: the row expires / is released
                return True
        return False

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        pass


def test_same_process_holder_is_preempted_instead_of_waited_on(monkeypatch):
    """A holder in this process cannot be another Hermes process: a second turn waiting on it
    must interrupt it (and stop its renewal) rather than wait out the 30-minute lease wait."""
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    first_agent = _agent(None)
    first_interrupts = []
    first_agent.interrupt = lambda msg, **kw: first_interrupts.append((msg, kw))
    holder = {"lease": None}
    db = _SameProcessDb(lambda: holder["lease"] if holder["lease"] and holder["lease"].is_turn_active() else None)
    first_agent._session_db = db
    first = _admit(first_agent).lease
    assert first is not None
    first.turn_active = True  # the first turn is running (admission leaves it inactive)
    holder["lease"] = first

    second_agent = _agent(db)
    second_agent.interrupt = lambda msg, **kw: None
    admission = _admit(second_agent)

    assert admission.early_result is None, "second turn must not time out behind its own process"
    assert admission.lease is not None
    assert first.stop.is_set() and first.is_turn_active() is False
    assert len(first_interrupts) == 1
    assert first_interrupts[0][0] == PREEMPT_MESSAGE
    assert any("previous turn" in s for s in second_agent.statuses)
    admission.lease.release()
    first.release()


def test_same_process_preempt_does_not_touch_the_waiting_agent_itself():
    """Re-entrant admission on the agent that already holds the lease must not preempt itself."""
    agent = _agent(_Db())
    interrupts = []
    agent.interrupt = lambda msg, **kw: interrupts.append(msg)
    lease = _admit(agent).lease
    assert lease is not None
    lease.turn_active = True
    assert preempt_live_turn_leases("s1", "x", exclude=agent) == 0
    assert interrupts == [] and lease.is_turn_active() is True
    lease.release()


def test_stop_preempts_superseded_same_process_turn_and_releases_registry():
    """/stop (no waiting agent to exclude) reaches a superseded turn that still holds the row."""
    agent = _agent(_Db())
    interrupts = []
    agent.interrupt = lambda msg, **kw: interrupts.append((msg, kw.get("tool_reason")))
    lease = _admit(agent).lease
    lease.turn_active = True
    assert preempt_live_turn_leases("s1", "stop") == 1
    assert interrupts and interrupts[0][0] == "stop"
    assert lease.stop.is_set() and lease.is_turn_active() is False
    # Idempotent: an already-winding-down turn is not interrupted twice.
    assert preempt_live_turn_leases("s1", "stop") == 0
    assert len(interrupts) == 1
    lease.release()
    # Released leases leave the process registry.
    assert preempt_live_turn_leases("s1", "stop") == 0
