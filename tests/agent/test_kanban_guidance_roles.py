"""Kanban tool access must not impersonate a dispatcher-owned worker."""
from types import SimpleNamespace


def test_orchestrator_guidance_does_not_claim_worker_ownership():
    from agent.system_prompt import _tool_guidance_block

    agent = SimpleNamespace(valid_tool_names={"kanban_show", "kanban_list", "kanban_create"})
    guidance = _tool_guidance_block(agent)
    assert guidance is not None
    assert "You have been assigned ONE task" not in guidance
    assert "kanban_list" in guidance
    assert "task_id" in guidance


def test_worker_keeps_lifecycle_guidance():
    from agent.system_prompt import _tool_guidance_block
    from agent.prompt_builder import KANBAN_GUIDANCE

    agent = SimpleNamespace(valid_tool_names={"kanban_show", "kanban_complete"})
    assert _tool_guidance_block(agent) == KANBAN_GUIDANCE


def test_no_kanban_tools_means_no_kanban_guidance():
    from agent.system_prompt import _tool_guidance_block

    assert _tool_guidance_block(SimpleNamespace(valid_tool_names={"terminal"})) is None


def test_cached_guidance_stays_session_static():
    from agent.system_prompt import _tool_guidance_block
    from agent.prompt_builder import KANBAN_GUIDANCE

    agent = SimpleNamespace(valid_tool_names={"kanban_show", "kanban_list"},
                            _kanban_worker_guidance=KANBAN_GUIDANCE)
    assert _tool_guidance_block(agent) == KANBAN_GUIDANCE
    agent._kanban_worker_guidance = ""
    assert _tool_guidance_block(agent) is None
