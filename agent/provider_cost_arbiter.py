"""Cost-aware OpenRouter provider arbitration.

OpenRouter's own "cheapest" ranking keys on the *prompt* price only (see
``wiki/projects/hermes/openrouter-provider-routing-price-metric-2026-10-11.md``), which is the
wrong objective for a long cached agent session: cache reads and completion tokens are ignored
entirely. This module computes the expected cost of every eligible endpoint against the
*measured* token mix of recent requests for the current model, and rewrites the provider
``order`` only when the savings actually pay back the one-time cold-cache penalty of moving.

Design constraints:

* **Cheap on the hot path.** The per-request work is one small JSON read plus a couple of integer
  comparisons. Endpoint pricing is fetched from OpenRouter at most once per ``endpoint_ttl`` and
  cached in-process and on disk.
* **Never break a turn.** Every public entry point swallows its own exceptions and degrades to
  "no opinion" (``None``), which leaves the configured routing exactly as it was.
* **Hysteresis, not churn.** A switch requires a payback test, a cooldown, and a minimum relative
  margin. Evaluating every N requests is free; *switching* every N requests would cost more than
  it saves.
* **Shared across processes.** Kanban workers are separate agent processes, so observations live
  in a file-locked JSON store under ``<hermes_home>/state/`` rather than in process memory.

State files:

* ``<hermes_home>/state/provider_cost.json`` - rolling observations, counters, decisions.
* ``<hermes_home>/state/provider_endpoints_cache.json`` - endpoint pricing cache (TTL).
* ``<hermes_home>/logs/provider_arbiter.jsonl`` - append-only decision audit trail.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("agent.provider_cost_arbiter")

#: Defaults; every one is overridable from ``provider_routing.autoselect`` in config.yaml.
DEFAULTS: Dict[str, Any] = {
    "enabled": False,              # opt-in: this rewrites routing decisions
    "window": 200,                 # observations kept per model for the rolling mix
    "cadence": 5,                  # evaluate every N completed requests
    "cooldown": 30,                # min completed requests between actual switches
    "min_margin": 0.15,            # required relative saving vs the current provider
    "payback_horizon": 25,         # expected remaining requests used for the payback test
    "compaction_factor": 0.4,      # post-switch prompt size as a fraction of the current one
    "endpoint_ttl": 3600,          # seconds before endpoint pricing is refetched
    "min_context": 0,              # require endpoint context_length >= this (0 = ignore)
    "keep_fallbacks": 2,           # how many trailing providers to keep for failover
    "min_uptime": 95.0,            # reject endpoints below this 1-day uptime percent
    "request_timeout": 8.0,
    "compact_on_switch": True,     # request one compaction pass before the new provider runs
}

_LOCK = threading.RLock()
_CATALOG_MEMO: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
#: Models whose catalog fetch failed (local/custom models, transient outages). Holding the
#: failure for one TTL keeps a non-OpenRouter model from paying a network round-trip per
#: cadence, and keeps the audit log from repeating the same "no_endpoints" line.
_CATALOG_FAILED: Dict[str, float] = {}


# --------------------------------------------------------------------------------------
# paths / low-level io
# --------------------------------------------------------------------------------------

def _hermes_home() -> Path:
    with suppress(Exception):
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    return Path(os.path.expanduser("~/.hermes"))


def _state_dir() -> Path:
    d = _hermes_home() / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _logs_dir() -> Path:
    d = _hermes_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _STORE_PATH() -> Path:
    return _state_dir() / "provider_cost.json"


def _ENDPOINT_CACHE_PATH() -> Path:
    return _state_dir() / "provider_endpoints_cache.json"


def _AUDIT_PATH() -> Path:
    return _logs_dir() / "provider_arbiter.jsonl"


class _FileLock:
    """Best-effort advisory lock; a no-op when ``fcntl`` is unavailable."""

    def __init__(self, path: Path):
        self._path = path
        self._fh = None

    def __enter__(self):
        try:
            import fcntl  # POSIX only
        except Exception:
            return self
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "a+")
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            self._fh = None
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            with suppress(Exception):
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            with suppress(Exception):
                self._fh.close()
            self._fh = None
        return False


def _read_json(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _write_json_atomic(path: Path, payload: Any) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
        return True
    except Exception:
        logger.debug("provider arbiter: state write failed", exc_info=True)
        return False


def _audit(event: Dict[str, Any]) -> None:
    try:
        rec = {"ts": time.time(), **event}
        with open(_AUDIT_PATH(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        logger.debug("provider arbiter: audit write failed", exc_info=True)


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------

def load_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``provider_routing.autoselect`` merged over :data:`DEFAULTS` (never raises)."""
    settings = dict(DEFAULTS)
    block: Any = None
    try:
        if cfg is None:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
        pr = (cfg or {}).get("provider_routing") or {}
        if isinstance(pr, dict):
            block = pr.get("autoselect")
    except Exception:
        block = None
    if isinstance(block, dict):
        for key, value in block.items():
            if value is not None:
                settings[key] = value
    return settings


def _per_model_settings(model: str, cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``provider_routing.models.<model>.autoselect`` overlay for *model* (never raises)."""
    try:
        from hermes_constants import resolve_per_model_provider_routing
        pr = (cfg or {}).get("provider_routing") or {}
        entry = resolve_per_model_provider_routing(model, (pr or {}).get("models"))
        if isinstance(entry, dict) and isinstance(entry.get("autoselect"), dict):
            return dict(entry["autoselect"])
    except Exception:
        pass
    return {}


# --------------------------------------------------------------------------------------
# usage observation
# --------------------------------------------------------------------------------------

def _key(model: str) -> str:
    return (model or "").strip() or "unknown"


def observe(
    model: str,
    provider: str,
    *,
    uncached_input: int,
    cached_input: int,
    output: int,
    cost: Optional[float] = None,
    window: int = 200,
) -> None:
    """Append one completed request to the rolling mix for *model* (never raises)."""
    try:
        with _LOCK, _FileLock(_STORE_PATH()):
            store = _read_json(_STORE_PATH(), {})
            if not isinstance(store, dict):
                store = {}
            obs = store.setdefault("observations", {})
            rows = obs.get(_key(model))
            if not isinstance(rows, list):
                rows = []
            rows.append({
                "ts": round(time.time(), 3),
                "provider": provider or "",
                "u": max(0, int(uncached_input or 0)),
                "c": max(0, int(cached_input or 0)),
                "o": max(0, int(output or 0)),
                "cost": float(cost) if cost is not None else None,
            })
            keep = max(10, int(window or 200))
            if len(rows) > keep:
                rows = rows[-keep:]
            obs[_key(model)] = rows
            counters = store.setdefault("counters", {})
            bucket = counters.setdefault(_key(model), {"since_eval": 0, "since_switch": 0})
            bucket["since_eval"] = int(bucket.get("since_eval", 0)) + 1
            bucket["since_switch"] = int(bucket.get("since_switch", 0)) + 1
            store["version"] = 1
            _write_json_atomic(_STORE_PATH(), store)
    except Exception:
        logger.debug("provider arbiter: observe failed", exc_info=True)


def rolling_mix(model: str, window: int = 200) -> Optional[Dict[str, Any]]:
    """Mean ``uncached_in``/``cached_in``/``out`` per request for *model*, or None."""
    try:
        store = _read_json(_STORE_PATH(), {})
        rows = ((store or {}).get("observations") or {}).get(_key(model)) or []
        rows = [r for r in rows if isinstance(r, dict)][-max(10, int(window or 200)):]
        if not rows:
            return None
        n = float(len(rows))
        return {
            "n": n,
            "uncached_in": sum(float(r.get("u") or 0) for r in rows) / n,
            "cached_in": sum(float(r.get("c") or 0) for r in rows) / n,
            "out": sum(float(r.get("o") or 0) for r in rows) / n,
            "served": [r.get("provider") for r in rows if r.get("provider")],
        }
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# endpoint catalog
# --------------------------------------------------------------------------------------

def _openrouter_key() -> Optional[str]:
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    # Fall back to the home .env without importing the heavy config stack.
    for candidate in (_hermes_home() / ".env", Path.home() / ".env"):
        try:
            for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
        except Exception:
            continue
    return None


def _float_price(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _normalize_endpoint(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        pricing = raw.get("pricing") or {}
        prompt = _float_price(pricing.get("prompt"))
        completion = _float_price(pricing.get("completion"))
        if prompt is None or completion is None:
            return None
        cache_read = _float_price(pricing.get("input_cache_read"))
        tag = str(raw.get("tag") or "")
        slug = tag.split("/")[0].strip() or str(raw.get("provider_name") or "").strip()
        if not slug:
            return None
        discount = _float_price(pricing.get("discount")) or 0.0
        return {
            "slug": slug,
            "provider": str(raw.get("provider_name") or slug),
            "tag": tag,
            "quantization": str(raw.get("quantization") or ""),
            "prompt": prompt,
            "completion": completion,
            "cache_read": cache_read if cache_read is not None else prompt,
            "has_cache_price": cache_read is not None,
            "discount": discount,
            "context_length": int(raw.get("context_length") or 0),
            "uptime": _float_price(raw.get("uptime_last_1d")),
            "throughput": _float_price((raw.get("throughput_last_30m") or {}).get("p50")),
            "latency": _float_price((raw.get("latency_last_30m") or {}).get("p50")),
            "supported_parameters": list(raw.get("supported_parameters") or []),
        }
    except Exception:
        return None


def fetch_endpoints(model: str, *, ttl: int = 3600, timeout: float = 8.0) -> Optional[List[Dict[str, Any]]]:
    """Endpoint list for *model*, memoized in-process and on disk for *ttl* seconds."""
    if not model:
        return None
    now = time.time()
    ttl = max(1, int(ttl))
    memo = _CATALOG_MEMO.get(model)
    if memo and (now - memo[0]) < ttl:
        return memo[1]
    failed_at = _CATALOG_FAILED.get(model)
    if failed_at is not None and (now - failed_at) < ttl:
        return None

    cache = _read_json(_ENDPOINT_CACHE_PATH(), {})
    entries = (cache or {}).get("models") if isinstance(cache, dict) else None
    entry = (entries or {}).get(model) if isinstance(entries, dict) else None
    if isinstance(entry, dict) and (now - float(entry.get("fetched_at") or 0)) < ttl:
        endpoints = entry.get("endpoints")
        if isinstance(endpoints, list) and endpoints:
            _CATALOG_MEMO[model] = (float(entry.get("fetched_at") or now), endpoints)
            return endpoints

    key = _openrouter_key()
    if not key:
        logger.debug("provider arbiter: no OpenRouter key; endpoint catalog unavailable")
        _CATALOG_FAILED[model] = now
        return memo[1] if memo else None
    try:
        import requests
        resp = requests.get(
            "https://openrouter.ai/api/v1/models/%s/endpoints" % model,
            headers={"Authorization": "Bearer %s" % key},
            timeout=min(5.0, float(timeout) or 5.0),
        )
        resp.raise_for_status()
        payload = resp.json()
        raw_list = ((payload or {}).get("data") or {}).get("endpoints") or []
        endpoints = [e for e in (_normalize_endpoint(r) for r in raw_list) if e]
        if not endpoints:
            _CATALOG_FAILED[model] = now
            return memo[1] if memo else None
        _CATALOG_FAILED.pop(model, None)
        _CATALOG_MEMO[model] = (now, endpoints)
        cache = cache if isinstance(cache, dict) else {}
        models = cache.setdefault("models", {})
        models[model] = {"fetched_at": now, "endpoints": endpoints}
        _write_json_atomic(_ENDPOINT_CACHE_PATH(), cache)
        return endpoints
    except Exception:
        logger.debug("provider arbiter: endpoint fetch failed for %s", model, exc_info=True)
        _CATALOG_FAILED[model] = now
        return memo[1] if memo else None


# --------------------------------------------------------------------------------------
# cost model + ranking
# --------------------------------------------------------------------------------------

def expected_cost(endpoint: Dict[str, Any], mix: Dict[str, float]) -> float:
    """Expected dollars for one *steady-state* (cache-warm) request with the rolling *mix*.

    Three terms, weighted by the measured mix:
    ``uncached*prompt + cached*cache_read + output*completion``. The one-time cold-cache
    penalty of arriving on a new provider is accounted separately by :func:`switch_penalty`
    and amortized by the payback test in :func:`decide`.
    """
    uncached = float(mix.get("uncached_in") or 0.0)
    cached = float(mix.get("cached_in") or 0.0)
    output = float(mix.get("out") or 0.0)
    return uncached * endpoint["prompt"] + cached * endpoint["cache_read"] + output * endpoint["completion"]


def switch_penalty(endpoint: Dict[str, Any], mix: Dict[str, float], *, compaction_factor: float = 1.0) -> float:
    """One-time extra cost of the first request on *endpoint* with a cold cache.

    Equal to the cache-read discount lost on the whole (post-compaction) prompt.
    """
    factor = 1.0 if compaction_factor is None else max(0.0, float(compaction_factor))
    prompt_tokens = (float(mix.get("uncached_in") or 0.0) + float(mix.get("cached_in") or 0.0)) * factor
    return prompt_tokens * max(0.0, endpoint["prompt"] - endpoint["cache_read"])


def _slugify(value: Any) -> str:
    """OpenRouter accepts ``anthropic``/``Anthropic``; compare slug-cased."""
    return str(value or "").strip().lower().replace(" ", "-").replace("_", "-")


def _quant_ok(endpoint: Dict[str, Any], allowed: Sequence[str]) -> bool:
    if not allowed:
        return True
    return (endpoint.get("quantization") or "").lower() in {str(q).lower() for q in allowed}


def rank_candidates(
    endpoints: Sequence[Dict[str, Any]],
    mix: Dict[str, float],
    *,
    allowed_quantizations: Sequence[str] = (),
    min_context: int = 0,
    min_uptime: Optional[float] = None,
    required_parameters: Sequence[str] = ("tools",),
    only: Sequence[str] = (),
    ignore: Sequence[str] = (),
) -> List[Tuple[Dict[str, Any], float]]:
    """Eligible ``(endpoint, expected_cost)`` pairs, cheapest first.

    ``only``/``ignore`` mirror the agent's own provider allow/deny lists so a manual veto
    (``provider_routing.only`` / ``ignore``, or ``/model`` arguments) still wins over cost.
    """
    ranked: List[Tuple[Dict[str, Any], float]] = []
    allow = {_slugify(x) for x in only if x}
    deny = {_slugify(x) for x in ignore if x}
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        if not _quant_ok(endpoint, allowed_quantizations):
            continue
        if allow and endpoint["slug"] not in allow:
            continue
        if endpoint["slug"] in deny:
            continue
        if min_context and int(endpoint.get("context_length") or 0) < int(min_context):
            continue
        uptime = endpoint.get("uptime")
        if min_uptime is not None and isinstance(uptime, (int, float)) and float(uptime) < float(min_uptime):
            continue
        params = set(endpoint.get("supported_parameters") or [])
        if required_parameters and not set(required_parameters) <= params:
            continue
        ranked.append((endpoint, expected_cost(endpoint, mix)))
    ranked.sort(key=lambda pair: (pair[1], pair[0].get("latency") or 0))
    return ranked


# --------------------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------------------

@dataclass
class Decision:
    """One arbitration outcome (also the shape written to the audit log)."""

    action: str                    # "hold" | "switch" | "skip" | "no_data"
    reason: str
    model: str
    current: Optional[str] = None
    chosen: Optional[str] = None
    order: List[str] = field(default_factory=list)
    savings_per_request: float = 0.0
    switch_penalty: float = 0.0
    payback_requests: float = 0.0
    current_cost: float = 0.0
    chosen_cost: float = 0.0
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    compaction_requested: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action, "reason": self.reason, "model": self.model,
            "current": self.current, "chosen": self.chosen, "order": self.order,
            "savings_per_request": round(self.savings_per_request, 9),
            "switch_penalty": round(self.switch_penalty, 9),
            "payback_requests": round(self.payback_requests, 2),
            "current_cost": round(self.current_cost, 9),
            "chosen_cost": round(self.chosen_cost, 9),
            "candidates": self.candidates[:8],
            "compaction_requested": self.compaction_requested,
        }


def _store_decision(model: str, decision: Decision, *, switched: bool) -> None:
    try:
        with _LOCK, _FileLock(_STORE_PATH()):
            store = _read_json(_STORE_PATH(), {})
            if not isinstance(store, dict):
                store = {}
            store.setdefault("decisions", {})[_key(model)] = {
                "ts": time.time(), "action": decision.action, "chosen": decision.chosen,
                "order": decision.order, "payback_requests": round(decision.payback_requests, 2),
                "current_cost": decision.current_cost, "chosen_cost": decision.chosen_cost,
            }
            counters = store.setdefault("counters", {})
            bucket = counters.setdefault(_key(model), {"since_eval": 0, "since_switch": 0})
            bucket["since_eval"] = 0
            if switched:
                bucket["since_switch"] = 0
                bucket["switches"] = int(bucket.get("switches", 0) or 0) + 1
            store["version"] = 1
            _write_json_atomic(_STORE_PATH(), store)
    except Exception:
        logger.debug("provider arbiter: decision persist failed", exc_info=True)


def _counters(model: str) -> Dict[str, int]:
    store = _read_json(_STORE_PATH(), {})
    bucket = ((store or {}).get("counters") or {}).get(_key(model)) or {}
    return {
        "since_eval": int(bucket.get("since_eval", 0) or 0),
        "since_switch": int(bucket.get("since_switch", 0) or 0),
        "switches": int(bucket.get("switches", 0) or 0),
    }


def _current_provider(model: str, mix: Dict[str, Any], window: int) -> Optional[str]:
    """Most frequently served provider over the rolling window (the de-facto current)."""
    served = [p for p in (mix.get("served") or []) if p]
    if not served:
        return None
    tail = served[-max(10, int(window or 200)):]
    counts: Dict[str, int] = {}
    last_seen: Dict[str, int] = {}
    for index, name in enumerate(tail):
        counts[name] = counts.get(name, 0) + 1
        last_seen[name] = index
    return max(counts, key=lambda name: (counts[name], last_seen[name]))


def decide(
    agent: Any,
    *,
    settings: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Optional[Decision]:
    """Arbitrate the provider order for *agent*, or return None to leave routing untouched."""
    try:
        model = str(getattr(agent, "model", "") or "")
        if not model:
            return None
        cfg = None
        with suppress(Exception):
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
        opts = dict(settings) if settings else load_settings(cfg)
        opts.update(_per_model_settings(model, cfg))
        if not opts.get("enabled"):
            return None

        quantizations = [q for q in (getattr(agent, "provider_quantizations", None) or []) if q]

        mix = rolling_mix(model, opts.get("window", 200))
        if not mix or mix["n"] < 3:
            _audit({"event": "no_data", "model": model, "observations": int((mix or {}).get("n") or 0)})
            return None

        counters = _counters(model)
        if not force:
            if counters["since_eval"] < int(opts.get("cadence", 5)):
                return None
            # Cooldown only guards against re-switching; a provider that has never been
            # switched away from is the pre-existing pin, not churn we caused.
            if counters["switches"] > 0 and counters["since_switch"] < int(opts.get("cooldown", 30)):
                _audit({"event": "cooldown", "model": model, "since_switch": counters["since_switch"]})
                return None

        endpoints = fetch_endpoints(model, ttl=int(opts.get("endpoint_ttl", 3600)),
                                    timeout=float(opts.get("request_timeout", 8.0)))
        if not endpoints:
            _audit({"event": "no_endpoints", "model": model})
            return None

        comp = float(opts.get("compaction_factor", 1.0) or 1.0)
        only = [p for p in (getattr(agent, "providers_allowed", None) or []) if p]
        ignore = [p for p in (getattr(agent, "providers_ignored", None) or []) if p]
        ranked = rank_candidates(
            endpoints, mix,
            allowed_quantizations=quantizations,
            min_context=int(opts.get("min_context", 0) or 0),
            min_uptime=opts.get("min_uptime"),
            required_parameters=("tools",),
            only=only,
            ignore=ignore,
        )
        if not ranked:
            _audit({"event": "no_eligible", "model": model, "quantizations": quantizations})
            return None

        current_name = _current_provider(model, mix, opts.get("window", 200))
        chosen, chosen_cost = ranked[0]
        current_row = next((e for e, _ in ranked if e["provider"] == current_name), None)

        top = [{"provider": e["provider"], "slug": e["slug"], "quantization": e["quantization"],
                "cost": round(c, 9)} for e, c in ranked[:5]]

        if current_row is None:
            decision = Decision(
                action="switch", reason="current provider not eligible", model=model,
                current=current_name, chosen=chosen["provider"], order=[e["slug"] for e, _ in ranked],
                current_cost=0.0, chosen_cost=chosen_cost, candidates=top, compaction_requested=True,
            )
        else:
            current_cost = expected_cost(current_row, mix)
            savings = current_cost - chosen_cost
            if current_name == chosen["provider"] or savings <= 0:
                decision = Decision(
                    action="hold", reason="already cheapest", model=model, current=current_name,
                    chosen=current_name, order=[e["slug"] for e, _ in ranked],
                    current_cost=current_cost, chosen_cost=current_cost, candidates=top,
                )
            else:
                margin = savings / current_cost if current_cost > 0 else 0.0
                penalty = switch_penalty(chosen, mix, compaction_factor=comp)
                payback = penalty / savings if savings > 0 else float("inf")
                horizon = float(opts.get("payback_horizon", 25) or 0)
                if margin < float(opts.get("min_margin", 0.15)):
                    decision = Decision(
                        action="hold", reason="margin %.1f%% below threshold" % (margin * 100), model=model,
                        current=current_name, chosen=current_name,
                        order=[e["slug"] for e, _ in ranked], current_cost=current_cost,
                        chosen_cost=chosen_cost, savings_per_request=savings,
                        switch_penalty=penalty, payback_requests=payback, candidates=top,
                    )
                elif payback > horizon:
                    decision = Decision(
                        action="hold", reason="payback %.1f > horizon %.0f" % (payback, horizon), model=model,
                        current=current_name, chosen=current_name,
                        order=[e["slug"] for e, _ in ranked], current_cost=current_cost,
                        chosen_cost=chosen_cost, savings_per_request=savings,
                        switch_penalty=penalty, payback_requests=payback, candidates=top,
                    )
                else:
                    decision = Decision(
                        action="switch", reason="payback %.1f <= horizon %.0f" % (payback, horizon), model=model,
                        current=current_name, chosen=chosen["provider"],
                        order=[e["slug"] for e, _ in ranked], current_cost=current_cost,
                        chosen_cost=chosen_cost, savings_per_request=savings,
                        switch_penalty=penalty, payback_requests=payback, candidates=top,
                        compaction_requested=bool(opts.get("compact_on_switch", True)),
                    )

        switched = decision.action == "switch"
        # Keep a small failover tail behind the winner so one outage cannot stall the fleet.
        keep = max(1, int(opts.get("keep_fallbacks", 2) or 1)) + 1
        decision.order = decision.order[:keep]
        _store_decision(model, decision, switched=switched)
        _audit({"event": "decision", **decision.as_dict(),
                "mix": {k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in mix.items() if k != "served"}})
        return decision
    except Exception:
        logger.debug("provider arbiter: decide failed", exc_info=True)
        return None


def override_from_decision(decision: Optional[Decision]) -> Optional[Dict[str, Any]]:
    """Translate a :class:`Decision` into an OpenRouter provider-preferences patch."""
    if not decision or decision.action != "switch" or not decision.order:
        return None
    return {"order": list(decision.order), "allow_fallbacks": True}


# --------------------------------------------------------------------------------------
# switch-time compaction handshake
# --------------------------------------------------------------------------------------

def request_compaction(agent: Any, decision: Decision) -> None:
    """Flag *agent* so the loop compacts before the new provider's first request."""
    with suppress(Exception):
        agent._provider_switch_compaction_pending = {
            "requested_at": time.time(),
            "model": decision.model,
            "from": decision.current,
            "to": decision.chosen,
        }
    decision.compaction_requested = True
    _audit({"event": "compaction_requested", "model": decision.model,
            "from": decision.current, "to": decision.chosen})


def consume_compaction_request(agent: Any) -> Optional[Dict[str, Any]]:
    """Pop the pending compaction request (once) so it triggers exactly one pass."""
    try:
        pending = getattr(agent, "_provider_switch_compaction_pending", None)
        if isinstance(pending, dict):
            agent._provider_switch_compaction_pending = None
            return pending
    except Exception:
        pass
    return None


def peek_compaction_request(agent: Any) -> Optional[Dict[str, Any]]:
    try:
        pending = getattr(agent, "_provider_switch_compaction_pending", None)
        return pending if isinstance(pending, dict) else None
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# integration seams
# --------------------------------------------------------------------------------------

def observe_response_usage(agent: Any, usage: Dict[str, Any]) -> None:
    """Telemetry seam: record one completed request from a canonical usage dict."""
    try:
        cache_read = int(usage.get("cache_read_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
        output = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        observe(
            str(getattr(agent, "model", "") or ""),
            str(getattr(agent, "provider", "") or ""),
            uncached_input=max(0, prompt - cache_read),
            cached_input=max(0, cache_read),
            output=max(0, output),
            window=int(load_settings().get("window", 200)),
        )
    except Exception:
        logger.debug("provider arbiter: usage observation failed", exc_info=True)


def provider_overrides_for_agent(agent: Any) -> Optional[Dict[str, Any]]:
    """Routing seam: the provider-preferences patch to apply for *agent*, or None."""
    try:
        decision = decide(agent)
        if decision is None:
            return None
        if decision.action == "switch" and decision.compaction_requested:
            request_compaction(agent, decision)
        return override_from_decision(decision)
    except Exception:
        logger.debug("provider arbiter: override resolution failed", exc_info=True)
        return None


def pending_switch_compaction(agent: Any) -> bool:
    """Loop seam: True when a switch wants one compaction pass before the next request."""
    try:
        return isinstance(getattr(agent, "_provider_switch_compaction_pending", None), dict)
    except Exception:
        return False
