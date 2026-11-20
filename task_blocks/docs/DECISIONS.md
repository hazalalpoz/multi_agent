# Decisions

Each section states what was chosen, why, what was rejected and for what
reason, and what the choice costs. Where a decision was changed by evidence,
the evidence is named.

## 1. Framework: LangGraph

LangGraph was chosen because this system's shape is a graph with cycles and
shared state. The verifier to regenerate loop is a cycle with a termination
condition, and `crisis` set in one node changes which tools exist in another. A
framework with explicit state and conditional edges expresses that directly,
and the diagrams in `ARCHITECTURE.md` can be read off the code because every
box is a node.

Three features decided it: typed state with reducers, so that `messages`
accumulates while `draft` is replaced; a checkpointer that makes conversation
persistence a configuration choice rather than a feature to build; and
`create_react_agent` for the specialists, so the ReAct loop is not written by
hand twice.

| Rejected | Reason |
|---|---|
| LangChain alone | Chains are acyclic. The evaluator-optimizer loop would become a hand-written `while` with its own state dict and attempt counter, which is a worse version of what LangGraph provides. LangChain is still used underneath for model clients and tools, which is the right level to use it at |
| CrewAI | Built around role-playing agents that delegate to each other. The control flow here is not negotiation: the router makes a decision and the verifier has veto power. Expressing a hard veto in a collaboration framework means fighting it |
| AutoGen | Strong at multi-agent conversation, where the unit of work is a message between agents. The hard requirements here concern a single turn's state machine, and conversational delegation makes the number of model calls per turn difficult to bound, which matters on a free tier |
| AutoGPT | An autonomous goal-pursuing loop. The requirement here is the opposite: a short, bounded, auditable path with a fixed number of model calls |
| Haystack | Oriented around retrieval pipelines. Retrieval here is a single store lookup, and the complexity is in control flow and validation, where Haystack offers least |
| No framework | Viable, since the graph is eight nodes and could be a state machine in 300 lines. Rejected for the checkpointer and the ReAct loops, which would be real work to write and more to make correct |

The cost is real: LangGraph brings a large dependency tree, and `langmem`
0.0.x pulls `langchain-anthropic` transitively even though it is never
imported.

## 2. Inference engine: Ollama locally, OpenRouter remotely

Ollama runs the router, verifier and embeddings; OpenRouter runs generation.

The reasoning is per-role rather than global. Classification runs on every
request, so its marginal cost dominates: locally it is zero, and the text never
leaves the host, which matters when the text may be someone describing a
crisis. Generation is where model size shows, and a 1.7B model cannot do it
well.

Ollama specifically provides one command to pull and serve, an
OpenAI-compatible API, and schema-constrained JSON output through `format`,
which is what makes the router's structured output reliable. Measured
classification latency is 1.27s at p50, CPU-only, and varies with machine load
across runs.

| Rejected | Reason |
|---|---|
| llama.cpp directly | Ollama is llama.cpp with model management and an HTTP server. Using it raw means building both. Worth revisiting for tighter control over quantisation and sampling |
| vLLM | The right answer for serving many concurrent requests on a GPU, where continuous batching and paged attention are large wins. There is no NVIDIA GPU here, and a single-user system has no concurrency to batch, so it would be slower rather than faster |
| Transformers in-process | No server and no model management, and a model in the app process means the API cannot restart without reloading weights. It also couples the app image to the model |
| TGI | Similar to vLLM: built for throughput this system does not have |
| Everything remote | Simpler, but classification would then cost money and quota on every request, and every message would reach a third party. Rate limiting would take out routing as well as generation. The daily cap was hit during development, and local routing is why the system still classified turns |
| Everything local | No remote dependency and no quota, but a 1.7B model generating the user-facing reply is a worse product. On CPU-only hardware a model large enough to generate well would be far too slow |

The cost is that Ollama in Docker on macOS is CPU-only, with no Metal access,
so local inference is slower here than on a Linux host with a GPU. This is
acceptable because the local models are small and the work is classification.

## 3. Model routing for cost optimisation

| Role | Model | Engine | Reason |
|---|---|---|---|
| router | `qwen3:1.7b` | local | The highest-volume call. Reliable schema-constrained JSON, and hybrid reasoning that can be turned off, since a reasoning preamble is pure latency for a three-field object. Measured 93.9% routing accuracy |
| verifier | `qwen3:1.7b` | local | Same reasons. A judge that costs money per call would not survive contact with a free tier |
| therapist | `google/gemma-4-31b-it:free` | OpenRouter | Dense 31B, open weights, strong instruction-following for register |
| logical | `google/gemma-4-26b-a4b-it:free` | OpenRouter | Sparse MoE with about 4B active, so markedly faster per token, which suits the role that makes more tool calls per turn |
| fallback | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter | A different provider, so a rate limit on Google's endpoints does not take the fallback with it. Large enough to be a genuine second opinion on a rejected draft |
| embeddings | `nomic-embed-text` | local | 768 dimensions, runs locally, so memory retrieval needs no network |

Two of the four model calls in a passing turn are free and local. The two
paid-tier calls are the ones that need a large model, and no request pays for
classification or verification.

Two corrections were made during the work. The originally specified
`openai/gpt-oss-20b:free` does not exist on OpenRouter, where only a paid
`openai/gpt-oss-20b` is listed, so the logical role moved to
`gemma-4-26b-a4b-it:free` after checking the live catalogue. The therapist was
moved off Nemotron because the NVIDIA free endpoint logs prompts, which is the
wrong property for the emotional path; Nemotron became the fallback instead,
where it still sees some content. That residual trade-off is not solved: a
regenerated reply on a crisis turn reaches an endpoint that logs.

Nemotron 3 Ultra does not advertise `response_format`, so
`with_structured_output` is never called on it. Structured output is used only
on qwen, through Ollama's JSON-schema mode.
`ModelConfig.supports_structured_output` records this per slot, so the
constraint lives in configuration rather than in someone's memory.

## 4. Isolation: hardened Docker containers with a no-egress inference network

The deployment uses Docker Compose, with `app` as non-root on a read-only root
filesystem, all capabilities dropped, `no-new-privileges`, CPU and memory
limits, and the inference engine on an internal network with no route off the
host.

This was verified rather than asserted. Writing to `/app` returns "Read-only
file system"; `CapDrop` is `[ALL]`; the process runs as uid 10001; and the
backend network has zero default routes, with raw-IP HTTP and ICMP both failing
while the app reaches OpenRouter normally.

| Option | What it would add | Reason not chosen |
|---|---|---|
| Plain container, no hardening | Nothing | Root in a container with full capabilities and a writable filesystem is a soft target, and the hardening above costs nothing at runtime |
| gVisor (runsc) | A user-space kernel, so container syscalls never reach the host kernel directly, giving real protection against kernel-level escapes | Not available on Docker Desktop for macOS. The threat it addresses is untrusted code execution, which this system does not do. It is the first thing to add if the agent ever runs model-generated code |
| Docker Sandboxes / Hardened Desktop | Managed, policy-driven isolation with less configuration | Ties the deployment to Docker Desktop's licensing and feature set, while the same controls are expressible in plain Compose, which runs anywhere |
| microVM (Firecracker, Kata) | A separate kernel per workload, with VM-level isolation at container-like startup times | Significant operational complexity for a single-node coursework system, and no nested-virtualisation path on this host. The right answer for multi-tenant untrusted workloads |
| Full VM | The strongest boundary available | Minutes to boot, gigabytes of overhead, and a whole OS to patch. Disproportionate for two small services |
| Kubernetes | Orchestration, autoscaling, network policies, secrets management | There is one node and one user, so Kubernetes would add a control plane larger than the system it runs. Network policies would express what Compose networks already express |
| Google AX / agent substrate | Managed agent runtime, sandboxing and scaling | The assignment requires deploying it directly, and a managed substrate would hide the decisions being assessed while adding a cloud dependency |

The chosen option does not protect against everything. A compromised `app`
process can still reach OpenRouter and read `/data`. There is no syscall
filtering beyond the default seccomp profile, and no protection against a
container-escape kernel exploit, which is what gVisor or a microVM would add.
This is a solid configuration for code written in-house and would be
insufficient for code that was not.

## 5. Verification layer: deterministic rules first, then a model

The verifier is hybrid. Rules in code run first, a local model then judges what
rules cannot see, and rule failures override the model in both directions.

| Rejected | Reason |
|---|---|
| No verification | The failure modes here are not crashes but plausible-sounding bad answers. Nothing else in the system can catch a crisis reply with no help resources |
| Model-only (LLM-as-judge) | Unfalsifiable and untestable. A 1.7B judge asked whether something is safe produces an opinion rather than a check, and cannot be unit tested, so a broken judge is invisible |
| Rules only | Rules cannot tell whether a reply answers the question, which is the most common quality failure |
| Self-critique in the same call | The model has already committed to its answer, and the critique comes from the process that produced the mistake |
| A large remote judge | Doubles remote cost and quota per turn, and makes validation depend on a third party being available. The quota exhaustion during development would have disabled verification entirely |

Measured precision and recall are both 100% on 11 hand-labelled drafts: 7
genuinely bad ones caught, and 4 acceptable ones, including a one-word answer
and an honest "I don't know", not rejected. The sample is small and the set is
ours, so this is evidence that the rules work as specified rather than a
general claim.

The cost is one extra local model call per turn, around 0.5 to 1 second. Rules
are also brittle in the usual way:
`test_emotional_turn_answered_as_a_bare_list_is_rejected` caught a regex that
could never match more than one line, which means that rule had been silently
dead since it was written.

## 6. Memory

Covered in full in [MEMORY.md](MEMORY.md), including why plain RAG is not
appropriate here and the comparison of upgraded memory systems. In summary,
short-term memory is LangGraph's `SqliteSaver` checkpointer keyed by
`thread_id`, and long-term memory is `SqliteStore` with a `sqlite-vec` index
over local embeddings, namespaced per `user_id` and written only by an agent's
explicit tool call.

## 7. Observability

Covered in full in [OBSERVABILITY.md](OBSERVABILITY.md), including the stack
comparison. In summary, Phoenix is used for traces because it understands LLM
spans natively and runs locally, Prometheus for metrics because the
agent-specific signals are counters and histograms, structured JSON logs are
correlated by `trace_id`, and Alertmanager delivers to a local webhook receiver
so the alert path is demonstrable without external accounts.

## 8. Alert destination: a local webhook receiver

A local receiver was chosen over Telegram and email. The alerting path is then
self-contained, with no external account and no credential in the repository,
and `GET /alerts` shows what has fired. It was verified end to end: `AppDown`
went from pending to firing, through Alertmanager to the receiver, and the
resolved notification arrived after recovery.

Telegram or SMTP is a receiver block in
[`ops/alertmanager/alertmanager.yml`](../ops/alertmanager/alertmanager.yml) and
nothing else changes. The cost is that nobody is actually woken up, which is
acceptable for coursework and would not be in production.

## 9. Crisis help resources: 112 and findahelpline.com only

This was changed by evidence. The first version included Tatarstan-specific
helplines found in secondary sources. The Tatarstan Ministry of Health page
that would have confirmed them was unreachable, so the numbers could not be
verified and were removed.

A wrong phone number in a crisis reply is the worst possible output of this
system, worse than no number, because it is trusted and wastes the one moment
that matters. The verifier rule accepts only these two resources, so a model
that invents a plausible-looking helpline still fails verification. There is a
test for exactly that case.

## 10. No LLM-as-judge in evals

Evaluation uses rule and reference checks, plus a manual rubric sheet for the
three cases where only a human can decide.

A judge model would make eval results depend on a third party's model version
and on free-tier quota, and an unvalidated judge produces numbers that look
rigorous while meaning little. The rubric sheet is emitted into the report with
the replies in full, so a human verdict is cheap to give and visibly a human
verdict. `compare_models.py` prints candidate replies side by side for the same
reason.

The cost is that answer quality has no single number, which is preferable to
the alternative.

## 11. Decisions driven by measurement

These were not chosen up front.

A second crisis detector was added after eval case `crisis-07`, a lethality
question behind a fictional frame, was missed by both the router model and the
intent keyword net and routed to the factual agent with no crisis skill loaded.
A separate method-request net now catches that class, with a
population-statistics exclusion so that "how many people die in traffic
accidents each year" remains a normal question. Crisis recall went from 6/7 to
7/7.

A tracing startup probe was added after Phoenix was found returning 401 on
every span push while the application logged `tracing_enabled`, because recent
images enable auth by default and OTLP export is fire-and-forget. Observability
that reports itself healthy while dropping everything is worse than none, so
there is now a probe that logs `tracing_collector_rejects_auth` at ERROR.

The retrieved-memory prompt wording was changed after models denied knowing
facts that were in their prompt when asked directly. The block now states that
a search has already run and that the facts are known. This is not fully fixed;
see [EVALS.md](EVALS.md).

`evals/results/` is tracked rather than ignored, because the report cites these
numbers and the run output is the evidence behind them.
