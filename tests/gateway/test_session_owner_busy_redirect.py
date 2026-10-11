from types import SimpleNamespace

from gateway.platforms.base import BasePlatformAdapter


def _store(mapping):
    return SimpleNamespace(_entries={k: SimpleNamespace(session_id=v) for k, v in mapping.items()})


def test_second_routing_key_on_same_session_id_routes_to_live_owner():
    fake = SimpleNamespace(
        _session_store=_store({"thread:A": "s1", "thread:B": "s1"}),
        _active_sessions={"thread:A": object()},
    )
    assert BasePlatformAdapter._active_owner_for_session(fake, "thread:B") == "thread:A"


def test_unrelated_session_is_not_redirected():
    fake = SimpleNamespace(
        _session_store=_store({"thread:A": "s1", "thread:C": "s2"}),
        _active_sessions={"thread:A": object()},
    )
    assert BasePlatformAdapter._active_owner_for_session(fake, "thread:C") is None


def test_missing_store_returns_none():
    fake = SimpleNamespace(_active_sessions={"thread:A": object()})
    assert BasePlatformAdapter._active_owner_for_session(fake, "thread:B") is None
