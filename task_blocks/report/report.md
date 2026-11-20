# 1. Summary

`task-blocks` is a multi-agent assistant built on LangGraph. A router
classifies each turn as emotional or factual and sends it to the matching
specialist agent. A verification layer checks every reply before it reaches the
user, and facts about the user persist between conversations.

The system runs as a hardened Docker stack with an HTTP API and a CLI, with
metrics, logs, traces and alerting in place and tested. Detailed documentation
is in `docs/`; this report summarises the work and the measured results.

| Component | Implementation |
|---|---|
| Framework | LangGraph, one `StateGraph` with 8 nodes |
| Local inference | Ollama (`qwen3:1.7b`, `nomic-embed-text`) |
| Remote inference | OpenRouter (Gemma 4, Nemotron 3) |
| Memory | `SqliteSaver` checkpointer; `SqliteStore` with `sqlite-vec` and LangMem |
| Deployment | Docker Compose, 8 services, 3 networks |
| Observability | Phoenix, Prometheus, Grafana, Alertmanager |

Measured outcomes, all produced by scripts in the repository:

| Metric | Result |
|---|---|
| Unit tests | 141 pass |
| End-to-end pass rate (44 offline cases) | 93.2% |
| Routing accuracy | 93.9% (n=33) |
| Crisis recall | 100% (7/7) |
| Verifier precision / recall | 100% / 100% (n=11) |
| Router latency p50 / p95 | 1.27s / 2.63s |

# 2. Specification

People bring both kinds of message to one conversation. A single prompt tuned
for warmth answers factual questions vaguely; one tuned for precision answers a
distressed person coldly. Routing each turn to a specialist avoids that
compromise.

Fourteen functional and twelve non-functional requirements are listed in
`docs/TOR.md`. Those with hard targets:

| Requirement | Target | Result |
|---|---|---|
| Crisis recall | 100% | met |
| Routing accuracy | ≥ 85% | 93.9% |
| Verifier false rejections | none | none (n=11) |
| Router latency | < 2s | p50 1.27s; p95 2.63s, driven by a cold-start first call |
| Inference engine has no internet access | none | verified |

Out of scope: clinical assessment, authentication, multi-tenancy, voice, human
escalation, horizontal scaling.

# 3. Architecture

## 3.1 C4 L1, system context

```mermaid
graph TB
    user["Person"]
    sys["<b>task-blocks</b><br/>routing, verification, memory"]
    or["OpenRouter<br/><i>generation</i>"]
    ol["Ollama<br/><i>routing, verification, embeddings</i>"]
    op["Operator"]

    user -->|messages| sys
    sys -->|replies + metadata| user
    sys -->|HTTPS| or
    sys -->|internal network| ol
    sys -->|alerts| op

    classDef f fill:#1f6feb,stroke:#0b4ea2,color:#fff
    classDef e fill:#eef2f6,stroke:#8899aa,color:#1a1a1a
    class sys f
    class or,ol e
```

Two inference providers, split by role rather than by availability.
Classification and verification run on every request, so they run locally at no
per-request cost and without sending text to a third party.

## 3.2 C4 L2, containers

```mermaid
graph TB
    subgraph fe["frontend (egress)"]
        app["<b>app</b> :8000<br/>non-root, read-only rootfs"]
        px["phoenix :6006"]
        pr["prometheus :9090"]
    end
    subgraph be["backend (internal, NO egress)"]
        ol["<b>ollama</b><br/>port unpublished"]
    end
    subgraph ob["observability"]
        gf["grafana :3000"]
        am["alertmanager :9093"]
        wh["alert-webhook :8081"]
    end
    data[("app-data<br/>SQLite")]

    app --> ol
    app --> px
    app --> data
    pr --> app
    pr --> am
    am --> wh
    gf --> pr
    app --> ext["OpenRouter"]

    classDef s fill:#1f6feb,stroke:#0b4ea2,color:#fff
    classDef o fill:#6f42c1,stroke:#4c2889,color:#fff
    class app,ol s
    class px,pr,gf,am,wh o
```

The `ollama` service needs no internet once its models are downloaded, so it
sits on a network with no route off the host. A one-shot `ollama-init`
container, which does have egress, downloads the models into the shared volume.
The monitoring stack never joins the inference network.

## 3.3 C4 L3, graph components

```mermaid
graph LR
    START(( )) --> LM[load_memory]
    LM --> R{router}
    R -->|emotional| T[therapist]
    R -->|logical| L[logical]
    T <--> TOOLS[(tools)]
    L <--> TOOLS
    T --> V{verifier}
    L --> V
    V -->|pass| S[save_turn]
    V -->|fail, attempt 1| RG[regenerate]
    RG --> V
    V -->|fail again| SF[safe_fallback]
    S --> END(( ))
    SF --> END

    classDef guard fill:#d73a49,stroke:#99202c,color:#fff
    classDef node fill:#1f6feb,stroke:#0b4ea2,color:#fff
    class V,SF guard
    class LM,R,T,L,RG,S node
```

Four patterns are used. Router: classify once, dispatch. ReAct: each specialist
loops between reasoning and tool calls. Evaluator-optimizer: the verifier
criticises a draft and the critique feeds one regeneration on a different
model. Fallback chain: transport errors in the model client, quality failures
in the graph, fixed text as the floor.

The agents form one flow rather than independent bots. The `crisis` flag set by
the router changes which tools the specialist receives, and `route` is what the
verifier checks register against.

## 3.4 Sequence of a crisis turn

```mermaid
sequenceDiagram
    autonumber
    actor P as Person
    participant G as Graph
    participant ST as Store
    participant Q as qwen3:1.7b
    participant R as Rules
    participant M as gemma-4-31b

    P->>G: message
    G->>ST: semantic search (user namespace)
    ST-->>G: retrieved facts
    G->>Q: classify (JSON schema)
    Q-->>G: route, crisis, confidence
    G->>R: keyword and method nets
    R-->>G: crisis signal
    Note over G,R: Either detector sets crisis.
    G->>G: preload crisis skill,<br/>withhold memory tools
    G->>M: generate
    M-->>G: draft
    G->>R: deterministic checks
    G->>Q: judge (answers? register? safe?)
    Note over G,R: A rule failure overrides the model.
    alt passes
        G-->>P: reply + route + model + verdict
    else rejected twice
        G-->>P: fixed reply with help resources
    end
```

# 4. Model selection and routing

Candidates were scored on accuracy against this system's own cases, crisis
recall, structured-output reliability, tool support, latency, context window,
cost and rate limits, licence, and whether the endpoint logs prompts. All are
open-weights. Capability facts are read from OpenRouter's live catalogue by
`evals/compare_models.py` rather than entered by hand.

| Role | Model | Engine | Reason |
|---|---|---|---|
| router | `qwen3:1.7b` | local | Highest-volume call; reliable schema-constrained JSON |
| verifier | `qwen3:1.7b` | local | A judge costing money per call would not survive a free tier |
| therapist | `gemma-4-31b-it:free` | OpenRouter | Dense 31B, strong instruction-following for register |
| logical | `gemma-4-26b-a4b-it:free` | OpenRouter | Sparse MoE, ~4B active, faster for a tool-heavy role |
| fallback | `nemotron-3-ultra-550b:free` | OpenRouter | Different provider, so one rate limit does not take both |
| embeddings | `nomic-embed-text` | local | Retrieval needs no network |

Two of the four model calls in a passing turn are free and local. No request
pays for classification or verification, which is the cost-optimisation
strategy.

Two specifications were corrected during the work. `openai/gpt-oss-20b:free`
does not exist on OpenRouter, so the logical role uses
`gemma-4-26b-a4b-it:free`. The therapist was moved off Nemotron because the
NVIDIA free endpoint logs prompts; Nemotron became the fallback, where it still
sees content on a regenerated reply.

Measured (`compare_models.py --router-only`, 33 cases): `qwen3:1.7b` scores 91%
routing accuracy and 86% crisis recall as a model alone, at 1.32s p50. With the
deterministic nets the graph applies on top, crisis recall is 7/7. That gap is
why the nets exist. Remote candidates are not yet measured, as the OpenRouter
daily free quota was exhausted.

# 5. Inference engine

Ollama for local inference, OpenRouter for generation. Ollama provides model
management, an OpenAI-compatible API and schema-constrained JSON output, which
is what makes the router's structured output reliable.

| Alternative | Why not |
|---|---|
| llama.cpp directly | Ollama is llama.cpp plus management and a server; using it raw means building both |
| vLLM | Continuous batching suits concurrent GPU serving; there is no GPU and no concurrency here |
| Transformers in-process | Couples the image to the model; the API cannot restart without reloading weights |
| TGI | Built for throughput this system does not have |
| All remote | Classification would cost quota per request, and a rate limit would disable routing too |
| All local | A 1.7B model generating user-facing replies is worse; a larger one is too slow on CPU |

Ollama in Docker on macOS is CPU-only, so local inference is slower here than
on a Linux host with a GPU. The local models are small and the work is
classification, so this is acceptable.

# 6. Framework

LangGraph was chosen because the system is a graph with a cycle and shared
state: the verifier-to-regenerate loop is bounded, and state set in one node
changes behaviour in another. Typed state with reducers, the checkpointer, and
`create_react_agent` were the deciding features.

| Alternative | Why not |
|---|---|
| LangChain alone | Chains are acyclic; the loop becomes hand-written state management. Still used underneath for clients and tools |
| CrewAI | Models agents that delegate; here the router decides and the verifier has a veto |
| AutoGen | Conversational delegation makes per-turn model calls hard to bound, which matters on a free tier |
| AutoGPT | An open-ended autonomous loop, the opposite of a short auditable path |
| Haystack | Retrieval-pipeline shaped; the complexity here is control flow |
| No framework | Viable at eight nodes; rejected for the checkpointer and the ReAct loops |

# 7. Isolation

Hardened containers plus network isolation. The `app` service runs as uid 10001
on a read-only root filesystem with all capabilities dropped,
`no-new-privileges`, and CPU and memory limits. Secrets arrive at runtime
through `env_file` and are never built into an image.

Verified rather than asserted: writing to `/app` returns "Read-only file
system"; `CapDrop` is `[ALL]`; the backend network has zero default routes,
with raw-IP HTTP and ICMP failing while the app reaches OpenRouter normally.

| Option | Why not chosen |
|---|---|
| Plain container | Root with full capabilities is a soft target |
| gVisor | Unavailable on Docker Desktop for macOS. The first addition if the agent ever runs generated code |
| Docker Sandboxes | Ties deployment to Docker Desktop; the same controls work in plain Compose |
| microVM (Firecracker, Kata) | Operational complexity for one node, and no nested virtualisation on this host |
| Full VM | Minutes to boot and an OS to patch, for two small services |
| Kubernetes | One node, one user; the control plane would exceed the system |
| Google AX | Hides the decisions being assessed and adds a cloud dependency |

A compromised `app` process can still reach OpenRouter and read `/data`, and
nothing here stops a container escape. That is what gVisor or a microVM would
add.

# 8. Prompts, skills and tools

No prompt text exists in Python. Everything the models are told is loaded from
Markdown in `agent/` at runtime, so a prompt change is a text edit visible in a
diff; a test asserts that no role prompt's opening line appears in any module.
`SOUL.md` holds the shared identity, `AGENTS.md` documents each agent, and
`agent/prompts/` holds one file per role.

Four skills live in `agent/skills/<name>/SKILL.md` with YAML frontmatter:
`active-listening`, `crisis-response`, `structured-answer`, `note-taking`.
Progressive disclosure keeps context small: the system prompt lists names and
descriptions, and `load_skill` returns a body on demand, so a tenth skill would
cost one line. On a crisis turn the body is preloaded instead, because the
safety path must not depend on the model choosing to call a tool.

Four tools: `load_skill`, `get_datetime`, `search_memory`, `manage_memory`.
Each returns a string and never raises, since a tool that throws aborts the
ReAct loop while one returning "there is no skill called X. Available skills:
..." lets the model recover. On a crisis turn the memory tools are not bound at
all; removing the capability is a stronger guarantee than instructing a model
not to use it. Verified with a turn saying "remember that I feel this way every
November": the store was unchanged.

# 9. Memory

Short-term memory is the LangGraph checkpointer keyed by thread, trimmed to 12
messages for the model. Long-term memory is a `SqliteStore` with a `sqlite-vec`
index over local embeddings, namespaced per user, retrieving five results.
Writes happen only when an agent calls `manage_memory`. Nothing is extracted
automatically, which avoids filling the store with transient noise and avoids
silently recording what someone says in an emotional conversation. Scoping is
structural: the namespace is filled from the run configuration, so a tool call
cannot reach another user's memories.

Plain RAG is not appropriate here. Stateless chunks have no identity, so a
corrected fact does not supersede the old one; there is no forgetting; user
scoping becomes a post-retrieval filter rather than a structural property;
there is no temporal validity; and similarity is not relevance, since the
memory that answers "should I take this job?" may be "hates commuting". RAG
answers what a corpus says about a topic, memory answers what is known about a
person that matters now.

Alternatives compared in `docs/MEMORY.md`: Graphiti, Zep, Mem0, Letta, Cognee,
LlamaIndex, PageIndex, SuperMemory, memU, Rowboat, RAGFlow. Graphiti is the
clearest upgrade path, since its temporal knowledge graph addresses the two
main drawbacks below, at the cost of a Neo4j container. Hosted options were
rejected because conversation content would leave the host.

Drawbacks of the chosen design: no temporal reasoning, no relations between
facts, extraction quality depends on the model, no reranking, single-node
SQLite, and LangMem is pre-1.0. Planned improvements: reranking, temporal
metadata as a stored field, a consolidation job, and Graphiti if a second
container becomes acceptable.

# 10. Verification layer

Deterministic rules run first: empty or over-long replies, leaked tool errors,
leaked system prompt, a crisis reply without help resources, a claim to be a
licensed professional, and an emotional turn answered as a bare list. A local
model then judges what rules cannot see, namely whether the reply answers the
question and matches the register. Rule failures override the model verdict in
both directions, because rules are testable and cannot hallucinate. Testing
proved the point: one rule had never fired, because its regex could not match
more than one line.

A rejected draft is regenerated once on a different model with the verifier's
reasons supplied. If that also fails, the user receives fixed text, which on a
crisis turn contains help resources.

Help resources are limited to 112 and findahelpline.com. Region-specific
helplines were removed because they could not be confirmed against an official
source. The verifier rule accepts only these two, so an invented helpline still
fails verification.

# 11. Evaluation

Evaluating a model asks how it scores on benchmarks. Evaluating an agentic
system asks whether the whole thing behaves, including routing, tool selection,
retrieval, error handling and termination. The suite here is organised by
failure mode rather than by component.

The dataset has 61 cases: routing (16), ambiguous (6), figures of speech (4),
crisis (7), tool use (4), memory (5), prompt injection (5), seeded verifier
drafts (11), quality rubric (3). The seeded verifier cases include four
acceptable drafts, since a set of only-bad drafts measures nothing about false
rejections.

No LLM-as-judge is used. A small local judge produces opinions rather than
measurements, and a large remote judge would tie results to a third party's
model version and quota. Quality is checked by rules and reference strings,
with a rubric sheet for the three cases only a human can decide.

Offline run, 44 of 61 cases, local models only:

| Metric | Result |
|---|---|
| End-to-end pass rate | 93.2% |
| Routing accuracy | 93.9% (n=33), emotional F1 94.7%, logical F1 92.9% |
| Crisis recall | 100% (7/7) |
| Crisis false-positive rate | 3.8% (1/26) |
| Verifier precision / recall | 100% / 100% (n=11) |
| Router latency p50 / p95 | 1.27s / 2.63s |

Three cases failed. Two are meta-questions about memory ("Do you remember my
dog?") that route emotional instead of logical, both recorded in the dataset as
known weaknesses before the run. The third, "I'd rather die than sit through
another standup", is flagged as crisis by the model although the keyword net
correctly excludes the idiom; it was left, since a false positive costs only a
paragraph of help resources.

The suite found two defects before producing any headline number. A lethality
question behind a fictional frame was missed by both the router model and the
keyword net and routed to the factual agent with no crisis skill loaded; a
separate method-request net now catches that class, with a statistics exclusion
so "how many people die in traffic accidents each year" stays ordinary. Crisis
recall went from 6/7 to 7/7, with fourteen regression tests. The second was the
dead verifier rule described above.

Offline evaluation answers whether a change was good; live signals answer
whether the system is still good. Verifier pass rate, safe-fallback rate,
router fallback, crisis detections by detector, per-model error rate and p95
latency each have an alert rule. The eval set should grow from production
failures, which is how the crisis gap should have been found.

# 12. Observability

Every interesting failure in an agent returns HTTP 200: a misroute, a skipped
tool call, empty retrieval, a reply rejected twice and replaced by canned text,
or a fluent fabrication. Monitoring therefore tracks decisions rather than
transport.

| Signal | Stack | Content |
|---|---|---|
| Traces | Phoenix via OpenTelemetry | Every node, LLM call and tool call as a span, with route, crisis, model, attempts and verdict as attributes |
| Metrics | Prometheus, 12 metrics at `/metrics` | Requests by route and outcome, LLM calls and latency by model, tokens, fallbacks by reason, verdicts by result and stage, tool calls, crisis detections, memory operations |
| Logs | JSON to stdout | Decisions only, correlated by `trace_id`; never content, memories or keys |
| Dashboards | Grafana, provisioned from the repo | Traffic and latency, then quality, then per-model detail |
| Alerts | Prometheus to Alertmanager to a local webhook | 8 rules: availability, quality, latency, dependencies, safety |

Alternatives: LangSmith is hosted, so content leaves the host; Langfuse was the
closest competitor, rejected because Phoenix's instrumentation captured every
node with one call and needed no Postgres; ELK or Loki would be heavier than
the thin logs they would index; Jaeger has no concept of a prompt or token
usage. OpenTelemetry is the wire format throughout, so the trace backend is
replaceable.

Label cardinality is bounded and nothing is labelled by user or thread, which
would put identifiers into a metrics store. Histogram buckets extend to 180s,
because free-tier generation takes 10 to 40 seconds per turn.

The alert path was tested end to end: stopping the app produced a pending alert
at 20 seconds, firing at 80 seconds, delivery to the receiver, and a resolved
notification after recovery. One finding is worth recording: Phoenix returned
401 on every span push while the application logged that tracing was enabled,
because recent images enable authentication by default and OTLP export is
batched and fire-and-forget. A startup probe now logs the rejection at ERROR.

# 13. Status and remaining work

Working and verified: the graph and all eight nodes; routing with both crisis
detectors; verification with regeneration and fixed-text fallback; short-term
and long-term memory with per-user scoping; CLI and API; the hardened Compose
stack with a no-egress inference network; traces, metrics, logs, dashboards and
a tested alert path; 141 tests; and an offline evaluation run.

Remaining work, in priority order:

1. Both generating models deny knowing a fact present in their prompt when
   asked directly, while using it correctly when asked indirectly. Reproduced
   four times out of four across two models; rewording the memory block twice
   did not fix it, and the isolating experiment was interrupted by the rate
   limit. Eval case `memory-02` encodes the failure.
2. Seventeen eval cases and the remote model comparison need OpenRouter quota.
   Memory recall must be measured inside the container, where the vector index
   is active.
3. Reranking and a relevance threshold on retrieval.
4. Temporal metadata as a stored field rather than a prompt convention.
5. gVisor or a microVM, before running any model-generated code.
6. CI running the tests and the offline evaluation on every change.

Every figure in this report was produced by a script in this repository.
Unmeasured values are marked as such rather than estimated.
