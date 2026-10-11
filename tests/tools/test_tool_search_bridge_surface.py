"""Bridge-only tool surface: keep_visible + defer_everything, session extras, skills index switch.

Synthetic tool definitions only; no dependency on the user's config.yaml.
"""

import json
import types

import pytest

from tools import tool_search as ts
from tools import tool_search_validation as tv
from tools.tool_search_session import set_session_extra_defs


def _schema(name, required=None):
    props = {k: {"type": "string"} for k in (required or [])}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} test tool",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(required or []),
            },
        },
    }


def _name(td):
    return (td.get("function") or td).get("name")


@pytest.fixture(autouse=True)
def _reset_session_extras():
    set_session_extra_defs(())
    yield
    set_session_extra_defs(())


@pytest.fixture
def bridge_cfg():
    return ts.ToolSearchConfig.from_raw({
        "defer_everything": True,
        "keep_visible": ["execute_code", "terminal"],
    })


def test_visible_surface_is_bridge_plus_keep_visible(bridge_cfg):
    defs = [_schema(n) for n in ("read_file", "terminal", "execute_code", "skill_view", "clarify")]
    set_session_extra_defs(defs)
    res = ts.assemble_tool_defs(defs, context_length=200_000, config=bridge_cfg)
    visible = {_name(td) for td in res.tool_defs}
    # Session extras are now ALWAYS visible (exempt from defer_everything), so all 5 test tools
    # plus bridge + keep_visible appear directly.
    assert visible == {"tool_search", "tool_describe", "tool_call", "terminal", "execute_code",
                       "read_file", "skill_view", "clarify"}
    assert res.deferred_count == 0


def test_defer_everything_defaults_off_for_plain_config():
    cfg = ts.ToolSearchConfig.from_raw({})
    assert cfg.defer_everything is False
    assert cfg.keep_visible == frozenset()


def test_unknown_name_is_not_deferrable_under_defer_everything(bridge_cfg):
    name, _args, err = ts.resolve_underlying_call({"name": "no_such_tool_xyz", "arguments": {}})
    assert name is None and err is not None


def test_session_extra_routes_through_resolve_validate_and_scope(bridge_cfg, monkeypatch):
    monkeypatch.setattr(ts, "load_config_readonly", lambda *a, **k: bridge_cfg)
    set_session_extra_defs([_schema("lcm_probe", required=["query"])])

    name, args, err = ts.resolve_underlying_call({"name": "lcm_probe", "arguments": {"query": "x"}})
    assert name == "lcm_probe" and err is None

    missing = tv.validate_deferred_call_args("lcm_probe", {})
    assert missing is not None and "query" in json.dumps(missing)
    assert tv.validate_deferred_call_args("lcm_probe", {"query": "x"}) is None

    scoped = ts.scoped_deferrable_names([_schema("read_file")])
    assert "lcm_probe" in scoped and "read_file" in scoped
    assert "terminal" not in scoped


def test_describe_finds_session_extra(bridge_cfg):
    set_session_extra_defs([_schema("lcm_probe", required=["query"])])
    out = json.loads(ts.dispatch_tool_describe(
        {"names": ["lcm_probe"]}, current_tool_defs=[_schema("read_file")], config=bridge_cfg))
    assert "lcm_probe" in out["tools"]


def test_describe_refuses_keep_visible_name(bridge_cfg):
    out = json.loads(ts.dispatch_tool_describe(
        {"names": ["terminal"]}, current_tool_defs=[_schema("terminal")], config=bridge_cfg))
    assert "terminal" in out.get("errors", {})


def test_apply_bridge_extras_is_idempotent(bridge_cfg, monkeypatch):
    import model_tools
    from agent import tool_bridge_extras as agent_init

    pre = [_schema("read_file"), _schema("terminal"), _schema("execute_code")]
    monkeypatch.setattr(model_tools, "get_tool_definitions",
                        lambda **k: list(pre))
    monkeypatch.setattr(model_tools, "_resolve_active_context_length", lambda: 200_000)
    # apply_bridge_extras resolves the policy through the platform-aware loader.
    monkeypatch.setattr(ts, "load_config", lambda *a, **k: bridge_cfg)
    monkeypatch.setattr(ts, "load_config_for_platform", lambda *a, **k: bridge_cfg)

    agent = types.SimpleNamespace(
        enabled_toolsets=None, disabled_toolsets=None, quiet_mode=True, side_agent=False,
        tools=list(pre) + [_schema("lcm_probe", required=["query"])],
        valid_tool_names=set(), _context_engine_tool_names={"lcm_probe"})

    agent_init.apply_bridge_extras(agent)
    first = sorted(_name(t) for t in agent.tools)
    # lcm_probe (session extra) is now always visible, so it appears in the direct tool list
    assert first == ["execute_code", "lcm_probe", "terminal", "tool_call", "tool_describe", "tool_search"]
    assert {"lcm_probe", "read_file", "terminal", "execute_code"} <= agent.valid_tool_names
    assert [_name(t) for t in agent._bridge_extra_defs] == ["lcm_probe"]

    agent_init.apply_bridge_extras(agent)
    assert sorted(_name(t) for t in agent.tools) == first
    assert [_name(t) for t in agent._bridge_extra_defs] == ["lcm_probe"]


def test_skills_index_switch(monkeypatch):
    from agent import system_prompt
    import hermes_cli.config as hc

    monkeypatch.setattr(hc, "load_config_readonly", lambda: {"skills": {"index_in_prompt": False}})
    assert system_prompt._skills_index_in_prompt() is False

    monkeypatch.setattr(hc, "load_config_readonly", lambda: {"skills": {"index_in_prompt": True}})
    assert system_prompt._skills_index_in_prompt() is True

    monkeypatch.setattr(hc, "load_config_readonly", lambda: {})
    assert system_prompt._skills_index_in_prompt() is True
