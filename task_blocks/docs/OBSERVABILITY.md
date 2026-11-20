# Observability

## Why HTTP metrics are not enough

A conventional service is observable through status codes and latency: a 500 is
a failure and a 200 is a success. An agent breaks that assumption, because
every failure mode below produces HTTP 200 and a plausible-looking reply:

- the router misclassified, so a distressed person received a correct, cold,
  factual answer;
- the model answered from assumption instead of calling the tool, and was
  wrong;
- retrieval returned nothing, so the reply is generic;
- the model denied knowing a fact that was in its prompt;
- a draft was rejected twice and the user received canned text;
- the reply is fluent and the facts in it are invented.

The signals therefore have to be about decisions rather than transport, which
is why `verifier_verdicts_total`, `crisis_detected_total{detector}` and
`fallback_total{reason}` exist, and why `agent_requests_total` is labelled by
outcome rather than status code.

Five further properties make agents awkward to monitor. Non-determinism means
the same input gives different output, so a single bad trace proves nothing and
rates over windows are the unit of measurement. Quality is invisible to
infrastructure, since no HTTP metric distinguishes a good answer from a bad
one, which is part of why having a verifier is valuable beyond safety: its pass
rate is the closest available proxy. Cost is per token rather than per request,
so two requests to the same endpoint can differ hundredfold and
`llm_tokens_total{model,type}` is the cost signal. Loops matter, because a
ReAct agent can call tools repeatedly and an evaluator-optimizer loop can
regenerate, so `attempts` and tool-call counts are recorded per turn. Finally,
prompt injection and drift both arrive as ordinary successful requests, and
injection shows up only if content is examined, which is why traces carry it
and logs do not.

## Stack comparison

| Option | Strengths | Reason not chosen as primary |
|---|---|---|
| Arize Phoenix (chosen, traces) | Understands LLM spans natively: prompts, completions, token counts, tool calls and nested agent steps, rendered as a readable tree. Self-hosted in one container, open source. Datasets and experiments can promote a real failure into a test case | Chosen. Weak at long-term aggregate metrics and has no alerting, hence Prometheus alongside it |
| LangSmith | The most polished tracing for LangChain and LangGraph, with excellent datasets and annotation queues | Hosted SaaS, so prompts and completions leave the host. For a system whose content may be someone describing a crisis, that is the wrong default. A self-hosted tier exists but is licensed |
| Langfuse | Open source, self-hostable, LLM-aware, with good cost tracking and prompt management | The closest competitor and a defensible alternative. Phoenix won because OpenInference's LangChain instrumentation was a single call that captured every node, LLM call and tool call with no per-call plumbing, and because it needed no Postgres |
| Langtrace | OpenTelemetry-native, self-hostable, vendor-neutral | Smaller ecosystem, and less mature LangGraph span semantics at the time of choosing |
| Prometheus + Grafana (chosen, metrics) | The standard for counters and histograms, with alert rules and dashboards as code in the repository, and no LLM-specific assumptions to fight | Chosen for metrics. Knows nothing about prompts or spans, so it cannot replace tracing |
| ELK / Loki | Powerful log search and aggregation | The logs here are deliberately thin, recording decisions rather than content, so a log search stack would be heavier than the thing it indexes. Loki would be reasonable if logs grew |
| OpenTelemetry + Jaeger | Vendor-neutral, mature distributed tracing, no lock-in | Jaeger renders a span tree but has no concept of a prompt, a completion or token usage, so the attributes that matter would be opaque strings. OpenTelemetry is used as the transport anyway, which makes Phoenix swappable |

The resulting shape is Phoenix for traces, Prometheus for metrics, structured
JSON logs to stdout, Grafana for dashboards, and Alertmanager to a local
webhook. OpenTelemetry is the wire format throughout, so any OTLP backend can
replace Phoenix without touching application code.

## Traces

Setup is `arize-phoenix-otel`'s `register()` plus OpenInference's
`LangChainInstrumentor`: one call, after which every LangGraph node, LLM call
and tool call becomes a span.

Agent-level attributes are attached with `set_span_attributes`, because a span
tree with timings but no decisions is not diagnostic:

```
task_blocks.route             emotional
task_blocks.crisis            False
task_blocks.confidence        0.9
task_blocks.model_used        safe-fallback
task_blocks.attempts          1
task_blocks.verdict           fail
task_blocks.verdict_source    rules
task_blocks.verdict_reasons   The reply is empty.
task_blocks.memories_retrieved 0
task_blocks.safe_fallback     True
```

That is a real capture, and it diagnoses one turn completely: retrieval found
nothing, generation produced an empty draft, a deterministic rule rejected it,
regeneration was attempted, that also failed, and the user received fixed text.
In measurement, 44 spans were produced for a handful of turns, with every graph
node present.

Tracing never causes an outage. If the collector is unreachable or
unconfigured, the app logs it and serves requests, since a missing
observability backend is not a production incident.

Silence is not success, however. Phoenix images now enable auth by default, and
OTLP export is batched and fire-and-forget, so the collector returned 401 on
every span push while the application logged `tracing_enabled`. Nothing
surfaced until the Phoenix container's own logs were read. There is now a
startup probe that posts to the collector and logs
`tracing_collector_rejects_auth` at ERROR if it is refused. Observability that
reports itself healthy while dropping everything is worse than none, because it
removes the suspicion that would have led to a check.

Traces contain message content, which is why Phoenix binds to `127.0.0.1` and
why logs do not.

## Metrics

`prometheus-client` exposes metrics at `GET /metrics`, scraped every 15
seconds. There are twelve, all defined in
[`observability.py`](../src/task_blocks/observability.py):

| Metric | Type | Labels |
|---|---|---|
| `agent_requests_total` | counter | `route`, `outcome` |
| `agent_request_latency_seconds` | histogram | `route` |
| `llm_calls_total` | counter | `model`, `role`, `status` |
| `llm_latency_seconds` | histogram | `model`, `role` |
| `llm_tokens_total` | counter | `model`, `type` |
| `fallback_total` | counter | `reason` |
| `verifier_verdicts_total` | counter | `result`, `source` |
| `tool_calls_total` | counter | `tool`, `status` |
| `crisis_detected_total` | counter | `detector` |
| `memory_ops_total` | counter | `op`, `status` |
| `router_confidence` | histogram | none |
| `task_blocks_build_info` | gauge | model IDs as labels |

Three design choices are worth defending. Label cardinality is bounded, since
`model` and `route` have a handful of values and nothing is labelled by
`user_id` or `thread_id`, which would be unbounded and would put identifiers
into a metrics store. Histogram buckets extend to 180 seconds, because
free-tier remote generation takes 10 to 40 seconds per turn and default buckets
would place nearly everything in `+Inf`, making p95 meaningless. Missing token
counts are skipped rather than recorded as zero, because free endpoints are
inconsistent about reporting usage, so the dashboards understate cost, which is
the safer direction.

Real values from one short run:

```
llm_calls_total{model="qwen3:1.7b",role="router",status="ok"} 3
llm_calls_total{model="google/gemma-4-31b-it:free",role="therapist",status="error"} 1
llm_tokens_total{model="google/gemma-4-31b-it:free",type="prompt"} 7659
verifier_verdicts_total{result="pass",source="llm"} 3
verifier_verdicts_total{result="fail",source="rules"} 1
fallback_total{reason="verifier_rejected"} 1
fallback_total{reason="safe_reply"} 1
```

Those seven lines describe a rate-limited turn: the therapist call errored, a
rule rejected the empty draft, regeneration ran, and the user received fixed
text. The roughly 3,300 prompt tokens per generation call are the cost of
carrying `SOUL.md`, the role prompt and the skill index in every system prompt,
which is the price of progressive disclosure implemented this way and is now
measurable per model.

## Logs

Logs are structured JSON to stdout, one object per line, correlated to traces
by `trace_id`:

```json
{"ts": "...", "level": "ERROR",
 "logger": "task_blocks.graph", "message": "node=therapist generation_failed",
 "trace_id": "d1538661635d1f4fe7c20c88c0533d35",
 "error": "OpenAIRateLimitError: Error code: 429 - Rate limit exceeded: free-models-per-day..."}
```

Message content, replies, memory contents and the API key are never logged.
Nodes log decisions such as route, model, verdict, attempt count and reason
counts, rather than what the user said. A therapy transcript in a log file is
the privacy failure this project is otherwise careful to avoid, so the verifier
logs `reason_count` rather than the reasons, which live in the trace instead.

Third-party HTTP loggers are held at WARNING, because they echo request URLs.
Container logs rotate at 10 MB across 3 files per service.

## Dashboard

One Grafana dashboard is provisioned from
[`ops/grafana/dashboards/task-blocks.json`](../ops/grafana/dashboards/task-blocks.json),
so panels are version controlled rather than assembled in the UI and lost on
restart.

It has three rows, ordered by what is looked at first. Traffic and latency
shows turns per minute by route and p50/p95 turn latency. Quality shows the
verifier pass rate as a thresholded stat, verdicts split by deciding stage,
fallbacks by reason, router confidence at p10 and p50, and crisis detections by
detector. Models, tools and memory shows LLM p95 latency and error rate by
model, tokens per minute by model and kind, tool and memory operations, and a
table of `task_blocks_build_info` showing which models the instance is actually
running.

The quality row sits above the per-model detail because for an agent the
verifier pass rate is closer to answering whether the system works than any
latency number.

## Alerting

Prometheus rules feed Alertmanager, which delivers to a local webhook receiver
that logs what it is sent and serves recent alerts at `GET /alerts`. This is
self-contained, with no external account and no credential in the repository.
Telegram or SMTP is a receiver block in
[`alertmanager.yml`](../ops/alertmanager/alertmanager.yml) and nothing else
changes.

There are eight rules in five groups:

| Alert | Condition | Severity |
|---|---|---|
| `AppDown` | not scrapeable for 1m | critical |
| `VerifierFailRateHigh` | rejections above 30% over 10m | warning |
| `SafeFallbackRateHigh` | canned replies above 5% over 15m | warning |
| `RouterFallbackActive` | structured output failing for 5m | warning |
| `TurnLatencyP95High` | p95 above 60s for 10m | warning |
| `RemoteModelErrorsHigh` | above 20% errors for a model over 10m | warning |
| `MemorySearchErrors` | any memory errors over 10m | warning |
| `CrisisDetectionRateUnusual` | sustained rise over 30m | info |

Thresholds come from observed behaviour, namely 10 to 40 second turns on
CPU-only local inference, and the comments in the rules file say so.
`repeat_interval` is 4 hours, because a quality regression does not need
re-notifying every five minutes and alert fatigue is the usual reason alerts
get ignored. An inhibition rule suppresses quality and latency warnings while
`AppDown` is firing, since they are consequences of it.

The last rule is informational rather than an error, because a sustained rise
in crisis-flagged turns is something a human should know about whether the
detector is misfiring or more people are genuinely in trouble.

The path was verified end to end. Stopping the app produced pending at t+20s,
firing at t+80s, Alertmanager showing the alert active, and the receiver
logging it with its summary and description. Restarting produced the resolved
notification five minutes later. `VerifierFailRateHigh` then went pending on
its own from the rate-limited turn, which validated two rules against real data
rather than by reading the YAML.

## Network and access

The monitoring stack sits on its own network and never joins the inference
network, since nothing in it has reason to talk to Ollama. Prometheus and
Phoenix also join `frontend` because they must reach the app, Prometheus to
scrape and the app to push spans.

Every UI binds to `127.0.0.1`. Grafana's anonymous access is off and its
credentials come from `.env`, but these services have no meaningful
authentication in this configuration and traces contain message content, so
loopback binding is the control that matters here. Exposing any of them would
require real authentication first.
