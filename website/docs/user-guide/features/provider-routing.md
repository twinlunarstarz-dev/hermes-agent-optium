---
title: Provider Routing
description: Configure OpenRouter provider preferences to optimize for cost, speed, or quality.
sidebar_label: Provider Routing
sidebar_position: 7
---

# Provider Routing

When using [OpenRouter](https://openrouter.ai) as your LLM provider, Hermes Agent supports **provider routing** — fine-grained control over which underlying AI providers handle your requests and how they're prioritized.

OpenRouter routes requests to many providers (e.g., Anthropic, Google, AWS Bedrock, Together AI). Provider routing lets you optimize for cost, speed, quality, or enforce specific provider requirements.

:::note
[Nous Portal](../../integrations/nous-portal.md) decides routing centrally per model and does not accept caller-supplied provider preferences; Hermes never sends the `provider` object to Portal, so `provider_routing` is simply ignored there.
:::

## Configuration

Add a `provider_routing` section to your `~/.hermes/config.yaml`:

```yaml
provider_routing:
  sort: "price"           # How to rank providers
  only: []                # Whitelist: only use these providers
  ignore: []              # Blacklist: never use these providers
  order: []               # Explicit provider priority order
  require_parameters: false  # Only use providers that support all parameters
  data_collection: null   # Control data collection ("allow" or "deny")
```

:::info
Provider routing only applies when using OpenRouter. It has no effect on Nous Portal or direct provider connections (e.g., connecting directly to the Anthropic API).
:::

## Options

### `sort`

Controls how OpenRouter ranks available providers for your request.

| Value | Description |
|-------|-------------|
| `"price"` | Cheapest provider first |
| `"throughput"` | Fastest tokens-per-second first |
| `"latency"` | Lowest time-to-first-token first |

```yaml
provider_routing:
  sort: "price"
```

### `only`

Whitelist of provider slugs. When set, **only** these providers will be used. All others are excluded. Use the lowercase slug shown by OpenRouter for each provider.

```yaml
provider_routing:
  only:
    - "anthropic"
    - "google"
```

### `ignore`

Blacklist of provider names. These providers will **never** be used, even if they offer the cheapest or fastest option.

```yaml
provider_routing:
  ignore:
    - "together"
    - "deepinfra"
```

### `order`

Explicit priority order. Providers listed first are preferred. Unlisted providers are used as fallbacks.

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
    - "amazon-bedrock"
```

### `require_parameters`

When `true`, OpenRouter will only route to providers that support **all** parameters in your request (like `temperature`, `top_p`, `tools`, etc.). This avoids silent parameter drops.

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection`

Controls whether providers can use your prompts for training. Options are `"allow"` or `"deny"`.

```yaml
provider_routing:
  data_collection: "deny"
```

### Per-model overrides (`models`)

Pin a different provider set per model. Keys under `models` are model ids; each entry takes the same
`sort` / `only` / `ignore` / `order` / `require_parameters` / `data_collection` keys and overrides the
flat value for that model only. Anything you don't set per model falls through to the flat defaults.

```yaml
provider_routing:
  sort: "price"                      # applies to every model
  models:
    "openai/gpt-6-astra":
      only: ["openai"]               # never let a reseller serve this one
    "anthropic/claude-fable-5.1":
      only: ["anthropic"]
    "moonshotai/kimi-k2.6":
      order: ["moonshotai", "together"]
      sort: "throughput"
```

Matching is spelling-tolerant like `agent.reasoning_overrides` (`claude-fable-5.1` / `claude-fable-5-1`,
with or without the `openrouter/` prefix). The override follows the model the agent is *currently* on, so
`/model` switches, fallback activation, cron jobs, and delegated subagents on another model each get their
own pins. Edit `config.yaml` directly for these keys: model ids contain dots, which `hermes config set`
reads as path separators.

## Practical Examples

### Optimize for Cost

Route to the cheapest available provider. Good for high-volume usage and development:

```yaml
provider_routing:
  sort: "price"
```

### Optimize for Speed

Prioritize low-latency providers for interactive use:

```yaml
provider_routing:
  sort: "latency"
```

### Optimize for Throughput

Best for long-form generation where tokens-per-second matters:

```yaml
provider_routing:
  sort: "throughput"
```

### Lock to Specific Providers

Ensure all requests go through a specific provider for consistency:

```yaml
provider_routing:
  only:
    - "anthropic"
```

### Avoid Specific Providers

Exclude providers you don't want to use (e.g., for data privacy):

```yaml
provider_routing:
  ignore:
    - "together"
    - "lepton"
  data_collection: "deny"
```

### Preferred Order with Fallbacks

Try your preferred providers first, fall back to others if unavailable:

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
  require_parameters: true
```

## How It Works

Provider routing preferences are passed to OpenRouter on agent chat requests and iteration-limit summaries via the `extra_body.provider` field. (`extra_body` is the OpenAI Python SDK argument; it becomes the top-level `provider` object in the JSON request.) Auxiliary tasks such as compression and title generation are configured independently under `auxiliary.<task>.extra_body`.

- **CLI mode** — configured in `~/.hermes/config.yaml`, loaded at startup
- **Gateway mode** — same config file, loaded when the gateway starts

The routing config is read from `config.yaml` and passed as parameters when creating the `AIAgent`:

```
providers_allowed  ← from provider_routing.only
providers_ignored  ← from provider_routing.ignore
providers_order    ← from provider_routing.order
provider_sort      ← from provider_routing.sort
provider_require_parameters ← from provider_routing.require_parameters
provider_data_collection    ← from provider_routing.data_collection
```

:::tip
You can combine multiple options. For example, sort by price but exclude certain providers and require parameter support:

```yaml
provider_routing:
  sort: "price"
  ignore: ["together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## Default Behavior

When no `provider_routing` section is configured (the default), the aggregator uses its own default routing logic, which generally balances cost and availability automatically.

:::tip Provider Routing vs. Fallback Models
Provider routing controls which **sub-providers behind OpenRouter** handle your requests. For automatic failover to an entirely different provider when your primary model fails, see [Fallback Providers](./fallback-providers.md).
:::

## Cost-Aware Auto-Selection (`autoselect`)

The static options above pick a provider using OpenRouter's own notion of "cheapest" — a
`sort: price` ranking that orders endpoints by **prompt price only**, ignoring cache-read
and completion pricing. That ranking is provably wrong for cache-heavy workloads: a
provider that wins on the prompt line can be several times more expensive per request once
a ~95% cache-hit rate on a large shared prefix and the output tokens are priced in.

`provider_routing.autoselect` replaces that guess with a measured one. It keeps a rolling
window of the **actual** token mix for each model, prices every eligible endpoint against
that mix, and switches only when the saving is real.

```yaml
provider_routing:
  sort: price
  models:
    deepseek/deepseek-v4.1-flash:
      order: [decart]          # the pin to beat
      allow_fallbacks: false
      quantizations: [fp4]     # never crossed by the arbiter
  autoselect:
    enabled: true
    window: 200             # completed requests kept in the rolling mix, per model
    cadence: 5              # evaluate every N completed requests
    cooldown: 30            # min requests between actual switches
    min_margin: 0.15        # required relative saving vs the provider in use
    payback_horizon: 25     # expected remaining requests used for the payback test
    compaction_factor: 0.4  # post-switch prompt size as a fraction of the current one
    endpoint_ttl: 3600      # seconds before endpoint pricing is refetched
    keep_fallbacks: 2       # trailing providers kept behind the winner for failover
    min_uptime: 95.0        # reject endpoints below this 1-day uptime percent
    compact_on_switch: true # one compaction pass before the new provider's first request
```

### The cost model

For each candidate endpoint the arbiter computes an expected cost per request from the
measured mix, not from a single price field:

```
E[cost] = E[uncached_input] x P_prompt
        + E[cached_input]   x P_cache_read
        + E[output]         x P_completion
```

`E[...]` are means over the rolling window; `P_*` come from the endpoint catalog. This is
the number that actually appears on the invoice, which is why it can rank differently from
OpenRouter's prompt-price sort.

### The switch decision

A switch costs a one-time cache rebuild: the new provider has none of your prefix cached,
so the first request after a move pays full prompt price for the whole context. Switching
for a small saving loses money. The arbiter therefore requires **all** of:

* `(cost_current - cost_best) / cost_current >= min_margin`
* `(cost_current - cost_best) x payback_horizon > switch_penalty`
* at least `cooldown` requests have passed since the last switch
* the endpoint is not quantized outside the requested class
* `uptime_1d >= min_uptime`

where `switch_penalty = (E[uncached_input] + E[cached_input]) x compaction_factor x
(P_prompt_best - P_cache_read_current)` — the cost of re-reading the live context uncached
once.

### Behaviour notes

* **Never crosses a quantization boundary.** With `quantizations: [fp4]` requested, only
  fp4 endpoints are ever considered, even if an fp8 endpoint is cheaper.
* **Quantization is part of the price identity.** `P_prompt` for `fp8` can be *lower* than
  for `fp4` while a different field is higher, so the same provider can rank differently
  depending on the requested class. The arbiter always prices the class in use.
* **One compaction pass per switch.** Because a move re-reads the whole prefix uncached,
  the loop performs a single compaction pass immediately before the new provider's first
  request, so it reads the smallest prompt it can. The request is consumed exactly once.
* **Churn guard.** The first-ever switch is exempt from `cooldown` so a mis-set config
  corrects itself on the next cadence; every switch after that waits.
* **Fail-open.** Any error (missing key, network failure, malformed catalog) leaves your
  configured routing in place. The arbiter never blocks a request and never raises into the
  turn loop. Models with no OpenRouter catalog are remembered as unavailable for one
  `endpoint_ttl` so a local model does not pay a fetch every cadence.

### State

| File | Contents |
|---|---|
| `$HERMES_HOME/state/provider_cost.json` | rolling mix + counters per model |
| `$HERMES_HOME/state/provider_endpoints_cache.json` | endpoint prices with TTL |
| `$HERMES_HOME/logs/provider_arbiter.jsonl` | append-only decision/switch audit trail |

Set `autoselect.enabled: false` (or delete the block) to disable. Cost is
`cost_current - cost_best <= 0` never switches.
