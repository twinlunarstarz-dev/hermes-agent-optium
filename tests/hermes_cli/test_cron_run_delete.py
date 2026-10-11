"""Dashboard cron run deletion removes every record the run-history list derives a run from."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import hermes_cli.web_routers.cron as _rt_cron


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    (default_home / "cron").mkdir(parents=True, exist_ok=True)
    (default_home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    from cron import executions as cron_executions

    monkeypatch.setattr(cron_executions, "EXECUTIONS_FILE", default_home / "cron" / "executions.db")
    monkeypatch.setattr(_rt_cron, "_job_owner_profile", lambda _job_id, _profile: "default")
    return default_home


def _at(name):
    return datetime.strptime(name, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=ZoneInfo("UTC"))


def _epoch(name):
    return _at(name).timestamp()


def _stub_job(monkeypatch, job_id, **extra):
    job = {"id": job_id, "name": job_id, "prompt": "", "no_agent": True, **extra}
    monkeypatch.setattr(
        _rt_cron,
        "_call_cron_for_profile",
        lambda _profile, cmd, *_a, **_k: dict(job) if cmd == "get_job" else None,
    )


def _ledger_attempt(monkeypatch, job_id, start_name, end_name, *, success=True, error=None):
    from cron import executions as cron_executions

    monkeypatch.setattr(cron_executions, "_hermes_now", lambda: _at(start_name))
    row = cron_executions.create_execution(job_id, source="direct")
    monkeypatch.setattr(cron_executions, "_hermes_now", lambda: _at(end_name))
    cron_executions.finish_execution(row["id"], success=success, error=error)
    return row["id"]


class _FakeSessionDB:
    def __init__(self, rows):
        self.rows = rows

    def list_cron_job_runs(self, job_id, limit=20, offset=0):
        return [dict(r) for r in self.rows if r.get("source") == "cron"][:limit]

    def get_session(self, session_id):
        row = next((r for r in self.rows if r["id"] == session_id), None)
        return dict(row) if row else None

    def delete_sessions(self, session_ids):
        before = len(self.rows)
        self.rows[:] = [r for r in self.rows if r["id"] not in set(session_ids)]
        return before - len(self.rows)

    def close(self):
        pass


def _install_session_db(monkeypatch, rows):
    db = _FakeSessionDB(rows)
    monkeypatch.setattr(_rt_cron, "_open_session_db_for_profile", lambda profile, **_k: db)
    return db


def _write_doc(home, job_id, name, text="output\n"):
    out = home / "cron" / "output" / job_id
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(text, encoding="utf-8")
    return out / name


def _run_ids(job_id):
    return [run["id"] for run in _rt_cron._list_cron_job_runs_sync(job_id, limit=50)["runs"]]


def test_deleting_script_only_run_removes_doc_and_ledger_so_it_cannot_reappear(isolated_home, monkeypatch):
    job_id = "job-del"
    _stub_job(monkeypatch, job_id)
    _install_session_db(monkeypatch, [])
    doc = _write_doc(isolated_home, job_id, "2026-09-25_10-00-00.md", "ok output\n")
    _ledger_attempt(monkeypatch, job_id, "2026-09-25_09-59-50", "2026-09-25_10-00-00")

    run_id = f"cron_output:{job_id}:2026-09-25_10-00-00"
    assert _run_ids(job_id) == [run_id]

    result = _rt_cron._delete_cron_job_runs_sync(job_id, run_id, None)

    assert result["deleted"] == 1
    assert result["skipped"] == []
    assert not doc.exists()
    assert _run_ids(job_id) == []


def test_deleting_session_run_takes_its_doc_and_ledger_but_keeps_neighbouring_run(isolated_home, monkeypatch):
    job_id = "job-del-session"
    _stub_job(monkeypatch, job_id)
    session_a = f"cron_{job_id}_a"
    session_b = f"cron_{job_id}_b"
    db = _install_session_db(monkeypatch, [
        {"id": session_a, "source": "cron", "started_at": _epoch("2026-09-25_10-00-00"),
         "ended_at": _epoch("2026-09-25_10-01-00"), "last_active": _epoch("2026-09-25_10-01-00")},
        {"id": session_b, "source": "cron", "started_at": _epoch("2026-09-25_10-05-00"),
         "ended_at": _epoch("2026-09-25_10-06-00"), "last_active": _epoch("2026-09-25_10-06-00")},
    ])
    doc_a = _write_doc(isolated_home, job_id, "2026-09-25_10-01-00.md")
    doc_b = _write_doc(isolated_home, job_id, "2026-09-25_10-06-00.md")
    ledger_a = _ledger_attempt(monkeypatch, job_id, "2026-09-25_10-00-00", "2026-09-25_10-01-00")
    ledger_b = _ledger_attempt(monkeypatch, job_id, "2026-09-25_10-05-00", "2026-09-25_10-06-00")

    assert _run_ids(job_id) == [session_b, session_a]

    result = _rt_cron._delete_cron_job_runs_sync(job_id, session_a, None)

    assert result["deleted"] == 1
    assert result["skipped"] == []
    assert [r["id"] for r in db.rows] == [session_b]
    assert not doc_a.exists()
    assert doc_b.exists()
    remaining = _rt_cron._list_cron_job_runs_sync(job_id, limit=50)["runs"]
    assert [run["id"] for run in remaining] == [session_b]
    from cron import executions as cron_executions

    assert [row["id"] for row in cron_executions.list_executions(job_id=job_id, limit=10)] == [ledger_b]
    assert cron_executions.get_execution(ledger_a) is None


def test_foreign_sessions_latest_row_and_traversal_ids_are_skipped_not_deleted(isolated_home, monkeypatch):
    job_id = "job-del"
    _stub_job(monkeypatch, job_id, last_run_at="2026-09-25T10:00:00+00:00", last_status="ok")
    db = _install_session_db(monkeypatch, [
        {"id": "cron_other-job_1", "source": "cron", "started_at": _epoch("2026-09-25_10-00-00"),
         "ended_at": _epoch("2026-09-25_10-01-00"), "last_active": _epoch("2026-09-25_10-01-00")},
        {"id": f"cron_{job_id}_cli", "source": "cli", "started_at": _epoch("2026-09-25_11-00-00"),
         "ended_at": _epoch("2026-09-25_11-01-00"), "last_active": _epoch("2026-09-25_11-01-00")},
    ])
    secret = isolated_home / "cron" / "secret.md"
    secret.write_text("keep me\n", encoding="utf-8")

    result = _rt_cron._delete_cron_job_runs_sync(
        job_id,
        ",".join([
            "cron_other-job_1",
            f"cron_{job_id}_cli",
            f"cron_output:{job_id}:latest",
            f"cron_output:{job_id}:../../secret",
        ]),
        None,
    )

    assert result["deleted"] == 0
    assert len(result["skipped"]) == 4
    assert len(db.rows) == 2
    assert secret.exists()


def test_running_ledger_attempts_are_never_deleted(isolated_home, monkeypatch):
    from cron import executions as cron_executions

    job_id = "job-del"
    _stub_job(monkeypatch, job_id)
    _install_session_db(monkeypatch, [])
    monkeypatch.setattr(cron_executions, "_hermes_now", lambda: _at("2026-09-25_12-00-00"))
    running = cron_executions.create_execution(job_id, source="direct")

    result = _rt_cron._delete_cron_job_runs_sync(job_id, f"cron_output:{job_id}:exec:{running['id']}", None)

    assert result["deleted"] == 0
    assert result["skipped"] and "not found" in result["skipped"][0]
    assert cron_executions.get_execution(running["id"]) is not None


def test_delete_route_takes_comma_separated_run_ids_and_rejects_oversized_requests(isolated_home, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    job_id = "job-del-route"
    _stub_job(monkeypatch, job_id)
    _install_session_db(monkeypatch, [])
    doc = _write_doc(isolated_home, job_id, "2026-09-25_10-00-00.md")
    _ledger_attempt(monkeypatch, job_id, "2026-09-25_09-59-50", "2026-09-25_10-00-00")
    app = FastAPI()
    app.include_router(_rt_cron.router)
    client = TestClient(app)

    response = client.delete(
        f"/api/cron/jobs/{job_id}/runs",
        params={"run_ids": f"cron_output:{job_id}:2026-09-25_10-00-00,cron_output:{job_id}:latest"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] == 1
    assert len(body["skipped"]) == 1
    assert not doc.exists()

    too_many = ",".join(f"cron_output:{job_id}:x{i}" for i in range(101))
    rejected = client.delete(f"/api/cron/jobs/{job_id}/runs", params={"run_ids": too_many})
    assert rejected.status_code == 400
