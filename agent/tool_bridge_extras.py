"""Route context-engine and memory-provider tools through the tool-search bridge.

Kept out of agent_init so it can be imported and tested without the agent bootstrap side effects.
"""


def apply_bridge_extras(agent) -> None:
    """Route context-engine and memory-provider tools through the tool-search bridge.

    Those tools are appended to ``agent.tools`` after registry assembly, so they would be
    sent on every request. Collect them as session extras, re-assemble from the full
    pre-assembly pool, and recompute ``valid_tool_names`` from that pool so ``tool_call``,
    the scope check and the execute_code sandbox all reach every tool. Idempotent: it runs
    after each injection step.
    """
    import model_tools
    from tools import tool_search as _ts
    from tools.tool_search_session import set_session_extra_defs
    from agent.oneshot_footprint import prune_oneshot_tools
    from tools.connectors.turn import side_agent_tool_drops

    bridge = set(_ts.BRIDGE_TOOL_NAMES)
    pre = model_tools.get_tool_definitions(
        enabled_toolsets=agent.enabled_toolsets, disabled_toolsets=agent.disabled_toolsets,
        quiet_mode=agent.quiet_mode, skip_tool_search_assembly=True) or []
    pre = prune_oneshot_tools(pre)
    drops = side_agent_tool_drops(agent)
    if drops:
        pre = [t for t in pre if t.get("function", {}).get("name") not in drops]
    pre_names = {t.get("function", {}).get("name") for t in pre}
    extras: dict = {}
    for t in list(getattr(agent, "_bridge_extra_defs", None) or []) + list(agent.tools or []):
        name = t.get("function", {}).get("name") if isinstance(t, dict) else None
        if name and name not in pre_names and name not in bridge and name not in extras:
            extras[name] = t
    extra_list = list(extras.values())
    agent._bridge_extra_defs = extra_list
    set_session_extra_defs(extra_list)
    full = pre + extra_list
    # Platform-aware: interactive Discord/CLI sessions keep the bridge-only surface here too.
    from tools.tool_search_session import session_platform as _session_platform
    assembly = _ts.assemble_tool_defs(
        full, context_length=model_tools._resolve_active_context_length(),
        config=_ts.load_config_for_platform(_session_platform() or getattr(agent, "platform", None)))
    agent.tools = assembly.tool_defs
    agent.valid_tool_names = {t.get("function", {}).get("name") for t in full} | bridge
