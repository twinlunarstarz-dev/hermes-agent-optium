"""Session extras for the tool-search bridge.

Tools the agent adds after registry assembly (context-engine tools such as LCM, and
memory-provider tools) are carried here so the bridge can list, describe, scope,
validate and dispatch them like registry tools. Stored in a ContextVar, which the agent
loop's thread propagation carries into tool worker threads.
"""

from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

_EXTRA_DEFS: ContextVar[Tuple[Dict[str, Any], ...]] = ContextVar(
    "hermes_tool_search_session_extra_defs", default=())


def set_session_extra_defs(defs: Any) -> None:
    """Install the current turn's extra tool schemas (``{"type": "function", ...}`` dicts)."""
    _EXTRA_DEFS.set(tuple(d for d in (defs or ()) if isinstance(d, dict)))


def session_extra_defs() -> List[Dict[str, Any]]:
    return list(_EXTRA_DEFS.get())


def session_extra_schema(name: str) -> Optional[Dict[str, Any]]:
    """Schema dict for an extra tool by name, or None."""
    for d in _EXTRA_DEFS.get():
        fn = d.get("function") if d.get("type") == "function" else d
        if isinstance(fn, dict) and fn.get("name") == name:
            return d

_PLATFORM: ContextVar[Optional[str]] = ContextVar(
    "hermes_tool_search_session_platform", default=None)


def set_session_platform(platform: Optional[str]) -> None:
    """Install the current turn's platform (e.g. ``"discord"``, ``"cli"``)."""
    _PLATFORM.set(str(platform) if platform else None)


def session_platform() -> Optional[str]:
    return _PLATFORM.get()
    return None
