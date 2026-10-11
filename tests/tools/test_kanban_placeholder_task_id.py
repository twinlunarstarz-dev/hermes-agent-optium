import pytest

from tools import kanban_tools as kt


@pytest.mark.parametrize("placeholder", [
    "HERMES_KANBAN_TASK", "$HERMES_KANBAN_TASK", "${HERMES_KANBAN_TASK}",
])
def test_env_var_name_is_not_a_task_id(monkeypatch, placeholder):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert kt._default_task_id(placeholder) is None


def test_real_task_id_passes_through():
    assert kt._default_task_id("t_abc123") == "t_abc123"
