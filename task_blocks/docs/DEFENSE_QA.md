# Defence notes

Answers to the questions likely to be asked. Where a number appears, it was
measured by a script in this repository.

## What is an LLM?

A neural network, in practice a transformer, trained on a very large amount of
text to predict the next token given the preceding ones. A token is a sub-word
chunk, so "unhappiness" might be three tokens. Training produces a set of
weights encoding statistical regularities of language, and with them a great
deal of implicit knowledge about the world.

Generation is sampling: the model emits a probability distribution over the
vocabulary, one token is chosen, it is appended to the input, and the process
repeats. That is the whole mechanism, and it explains most of the model's
properties, including its weaknesses, since there is no separate step that
checks whether what is being produced is true.

## What does the "L" in LLM stand for?

Large, describing the model rather than the language: large in parameters,
numbering in the billions, and in training data. `qwen3:1.7b` has about 1.7
billion parameters, while `nemotron-3-ultra-550b-a55b` has about 550 billion,
of which roughly 55 billion are active per token because it is a
mixture-of-experts model.

## What is an agent?

A system that uses an LLM to decide what to do next, not only what to say. The
model is the reasoning component inside a loop that can take actions, such as
calling tools, reading and writing memory, or handing off to another component,
then observe the results and continue until the task is done.

The distinguishing feature is agency over control flow. An LLM answers; an
agent decides.

## How do an LLM and an agent differ?

| | LLM | Agent |
|---|---|---|
| Unit of work | One completion | A task, over many steps |
| Control flow | The caller decides everything | The system decides, guided by the model |
| State | Only what is in the prompt | Explicit state, memory, checkpoints |
| Side effects | None | Tool calls, writes, external actions |
| Failure | A bad sentence | A bad sequence of actions |
| Observability | Input and output | Trajectory: every decision and call |

In this system the LLMs are interchangeable components, swapped by changing a
model ID in `config.py`. The agent is the graph: routing, tools, memory,
verification and the fallback chain. That is where the engineering is.

## What are hallucinations?

Output that is fluent, confident and false. It is not a bug added on top of the
mechanism but a consequence of it: the model samples plausible continuations,
and plausibility and truth come apart, most sharply for specifics such as
version numbers, citations, statistics and phone numbers.

What matters practically is that a hallucination looks exactly like a correct
answer from the outside, with no internal signal to read.

Since hallucination cannot be eliminated, this system responds in four ways. It
provides tools for anything the model cannot know, so `get_datetime` replaces
guessing a date. It stores facts about the person rather than inventing them.
Its prompts make "I don't know" an acceptable answer and give an explicit shape
for one. And its verifier flags invented specifics, with deterministic rules
for the cases where a hallucination would be dangerous: a crisis reply is
rejected unless it contains one of the two verified help resources, so an
invented helpline fails even though it looks right. There is a test for that
case.

## What is a context window, and what does it affect?

The maximum number of tokens the model can attend to at once, covering the
system prompt, conversation, retrieved memories, tool definitions, tool results
and the reply being generated. `qwen3:1.7b` has about 40k, the Gemma models
here about 262k, and Nemotron about 1M.

It affects three things. How much history fits, since beyond the window
something must be dropped or summarised, which is why history is trimmed to the
last 12 messages here. Cost and latency, since attention cost grows with
context and prompt tokens are billed; measured here at about 3,300 prompt
tokens per generation call, because every system prompt carries `SOUL.md`, the
role prompt and the skill index. And accuracy within the window, since a large
window does not mean uniform attention: material in the middle of a long
context is used less reliably than material at the ends, so a bigger window is
not a substitute for retrieving the right things.

This is why skills use progressive disclosure: agents see names and
descriptions and fetch a body only when needed, so a tenth skill costs one line
of prompt rather than a tenth of the window.

## What does the architecture look like, and which patterns does it use?

One `StateGraph`: `load_memory → router → (therapist | logical) → verifier →
save_turn`, with `verifier → regenerate → verifier` as a bounded cycle and a
fixed safe reply as the floor. Diagrams are in
[ARCHITECTURE.md](ARCHITECTURE.md).

Four patterns are used. Router: classify once, then dispatch to a specialist,
because one prompt covering both registers is mediocre at both. ReAct: each
specialist loops between reasoning and tool calls. Evaluator-optimizer: the
verifier criticises the draft and the critique becomes input to a regeneration
on a different model, with the critic a separate and partly deterministic
component so it cannot approve its own output. Fallback chain: transport
failures inside the model client, quality failures in the graph, and fixed text
at the end.

The agents are joined into one flow with explicit state rather than run as
independent bots. The `crisis` flag set by the router changes which tools the
specialist has, and `route` is what the verifier checks register against.

## Which frameworks were considered, and why LangGraph?

Considered: LangChain alone, CrewAI, AutoGen, AutoGPT, Haystack, and no
framework. Full reasoning is in [DECISIONS.md](DECISIONS.md).

In short, this system is a graph with a cycle and shared state. LangChain
chains are acyclic, so the regeneration loop would become a hand-written loop
with its own state. CrewAI and AutoGen model agents negotiating, but the
control flow here has a hard veto rather than a negotiation, and conversational
delegation makes the per-turn model-call count hard to bound, which matters on
a free tier. AutoGPT is an open-ended autonomous loop, the opposite of a short
auditable path. Haystack is shaped around retrieval pipelines, while the
complexity here is control flow. No framework was genuinely viable at eight
nodes, and was rejected for the checkpointer and the ReAct loops, which would
be real work to write correctly.

## How were the LLMs chosen, and which were tested?

By role, against this system's own cases, rather than by leaderboard.

Candidates were `qwen3:1.7b` (local), `google/gemma-4-31b-it:free`,
`google/gemma-4-26b-a4b-it:free` and `nvidia/nemotron-3-ultra-550b-a55b:free`,
all open-weights as required.

Criteria were accuracy on our cases, crisis recall, structured-output
reliability (since the router and verifier need schema-valid JSON every time),
tool-calling support, latency, context window, cost and rate limits, licence,
and privacy, meaning whether the endpoint logs prompts.

Measured so far, from `compare_models.py --router-only` over 33 cases,
`qwen3:1.7b` scores 91% routing accuracy and 86% crisis recall as a model
alone, at about 1.3s p50. The remote candidates are TBD (run
`evals/compare_models.py` when OpenRouter quota is available).

Two choices are worth defending. A 1.7B model handles routing because it is the
highest-volume call in the system and a two-way classification rather than open
generation; running it locally makes routing free per request and keeps the
text, which may be someone describing a crisis, on the host. The 86% model-only
crisis recall is also why the deterministic nets exist, since at node level
with both nets crisis recall is 7/7. And the therapist is not Nemotron because
the NVIDIA free endpoint logs prompts, which is the wrong property for the
emotional path, so Nemotron became the fallback and Gemma 4 31B the therapist;
the residual trade-off is that a regenerated crisis reply still reaches an
endpoint that logs.

Also worth mentioning: the originally specified `openai/gpt-oss-20b:free` does
not exist on OpenRouter. This was found by querying the live catalogue rather
than by trusting the specification, and the role moved to
`gemma-4-26b-a4b-it:free`.

## What are the memory types, and which does this system use?

Working, episodic, semantic and procedural. Full treatment is in
[MEMORY.md](MEMORY.md).

Working or short-term memory is the current conversation, held in a
`SqliteSaver` checkpointer keyed by `thread_id` and trimmed to 12 messages for
the model. Semantic memory holds durable facts per person in a `SqliteStore`
with a `sqlite-vec` index over local `nomic-embed-text` embeddings, namespace
`("memories", user_id)`. Episodic memory exists only partially, since
resumable threads hold past episodes but there is no event timeline, making it
the weakest of the four here. Procedural memory is the `agent/` Markdown files
holding identity, prompts and skills, versioned in the repository rather than
learned at runtime, so a human reviews any change to how the system behaves.

Short-term and long-term memory are handled differently. Short-term is
complete, chronological, automatic and scoped to one conversation. Long-term is
sparse, semantic, retrieved by similarity, scoped to a person, and written only
when an agent explicitly decides to, never automatically, because automatic
extraction with a small model fills the store with noise and silently recording
what someone says in an emotional conversation is the wrong default.

Plain RAG is not suitable because stateless chunks have no identity, so a
corrected fact does not supersede the old one; there is no forgetting; user
scoping is a filter rather than a structural property; there is no temporal
validity; and similarity is not relevance, since the memory that answers
"should I take this job?" may be "hates commuting", related by inference rather
than wording. RAG answers what a corpus says about a topic, while memory
answers what is known about this person that matters now.

## What are the tools?

Four, in [`tools.py`](../src/task_blocks/tools.py): `load_skill`,
`get_datetime`, `search_memory` and `manage_memory`, the last two from LangMem
over the long-term store.

The contract is that a tool returns a string and never raises, because a tool
that throws aborts the ReAct loop while one returning "there is no skill called
X. Available skills: …" lets the model recover by itself. Each has a docstring
the model uses to choose it, validated arguments, and errors phrased as
guidance; the `get_datetime` error names the IANA format, because a model
passing "Moscow" instead of "Europe/Moscow" is a common and recoverable
mistake.

One tool decision is worth highlighting. On a crisis turn the memory tools are
not bound at all, because the crisis skill forbids storing anything from such a
conversation and removing the capability is a stronger guarantee than
instructing a model not to use it. This was verified with a turn that
explicitly said "remember that I feel this way every November": the store was
unchanged.

## What are skills, and how do you build them correctly?

A skill is a unit of procedural knowledge in a file, with YAML frontmatter
giving `name` and `description`, followed by a Markdown body of instructions.
There are four: `active-listening`, `crisis-response`, `structured-answer` and
`note-taking`.

Progressive disclosure is the mechanism. The system prompt lists names and
descriptions only, and `load_skill(name)` returns the body, so a tenth skill
costs one line of context rather than a tenth of the window.

Building them correctly means several things. The description is a routing
decision, since it is all the model sees when choosing, so it must say when to
use the skill rather than only what it is about. The body should contain
instructions rather than encyclopaedia entries: "say back what they told you,
with the detail that makes it theirs" is actionable, while "reflective
listening is a technique in which…" is not. Counter-examples help, since the
failure mode is usually more specific than the success, so each skill here
shows weak against better phrasing. Hard boundaries belong in the file, in the
imperative: never minimise, never give methods, never save credentials. One
skill should do one job, since a skill covering both listening and crisis
response would be loaded in full for either. Where it matters, skills should be
testable: the crisis skill's help resources are mirrored by a deterministic
verifier rule and a test, so a reply that ignores the skill still fails. And
they should be versioned in the repository, since they are the system's
behaviour and changes should be reviewable in a diff.

A skill's body should not sit in the system prompt permanently, except where
safety outranks economy: on a crisis turn the body is preloaded, because the
safety path must not depend on the model choosing to call a tool.

## How are results evaluated, and what are the best practices?

At three levels, namely the model, the agentic system, and this system, with
the suite organised by failure mode rather than by component. Detail is in
[EVALS.md](EVALS.md).

There are 61 cases across routing, ambiguity, figures of speech, crisis, tool
use, memory, prompt injection, seeded verifier drafts and a manual quality
rubric. Measured so far: 141 unit tests passing, and an offline run at 93.2%
pass rate, 93.9% routing accuracy, 100% crisis recall, and verifier precision
and recall both 100% on 11 hand-labelled drafts.

The practices followed were: label by failure mode, so a regression names its
own cause; include true negatives, since the seeded verifier set has four
acceptable drafts and a set of only bad drafts measures nothing about false
rejections; have one hard gate, so `run_evals.py` exits non-zero if any crisis
case is missed whatever else passed; avoid LLM-as-judge, since a small judge
gives opinions and a large one ties results to a third party's model version
and quota, with the cost that answer quality has no single number; record the
environment, so every report is stamped with mode, target and whether vector
search was available; grow the set from real failures and keep known weaknesses
in the dataset as notes before they are fixed; and separate offline from
online, since offline evals answer whether a change was good while the live
verifier pass rate and fallback rate answer whether it is still good.

The strongest argument for having evals is that they found two real defects
before producing any headline number. A lethality question behind a fictional
frame was missed by both the router model and the keyword net and routed to the
factual agent with no crisis skill, which was fixed with 14 regression tests
and moved crisis recall from 6/7 to 7/7. And a verifier rule had never fired,
because its regex could not match more than one line.

## Why is isolation needed, and what kinds are there?

For three reasons here. An agent executes decisions made by a model influenced
by untrusted input, so prompt injection becomes an action risk rather than only
a text risk. It holds secrets, namely an API key and other people's personal
facts. And it reaches the network, so a compromise is an exfiltration path.

The kinds, from weakest to strongest, are process-level isolation through users
and permissions; containers using namespaces and cgroups, which share a kernel;
hardened containers adding non-root execution, a read-only filesystem, dropped
capabilities, seccomp and resource limits; sandboxed runtimes such as gVisor,
which provide a user-space kernel so syscalls do not reach the host kernel;
microVMs such as Firecracker or Kata, giving a separate kernel with
container-like startup; and full VMs, the strongest and heaviest. Network
isolation is orthogonal and applies at every level.

This system uses hardened containers plus network isolation. The `app` service
runs as uid 10001 on a read-only root filesystem with all capabilities dropped,
`no-new-privileges`, and CPU and memory limits. The inference engine sits on an
internal network with zero default routes, verified with raw-IP HTTP and ICMP
both failing, so even if a tool were induced to talk to it there would be
nowhere to send anything. Its port is not published, and secrets arrive at
runtime through `env_file` rather than being built into an image.

It does not protect against everything. A compromised `app` process can still
reach OpenRouter and read `/data`, and there is no protection against a
kernel-level container escape, which is what gVisor or a microVM would add and
is the first thing to add if this system ever executes model-generated code.
This is a solid configuration for code written in-house and insufficient for
code that was not.

## How is observability implemented, and where do alerts go?

Phoenix for traces, Prometheus and Grafana for metrics and dashboards,
structured JSON logs to stdout, and Alertmanager to a local webhook receiver.
OpenTelemetry is the wire format, so Phoenix is swappable. The comparison of
alternatives is in [OBSERVABILITY.md](OBSERVABILITY.md).

HTTP metrics alone are insufficient because every interesting failure returns
200: a misroute, a skipped tool call, empty retrieval, a double rejection
ending in canned text, or a fluent fabrication are all successful requests. The
signals are therefore about decisions: `verifier_verdicts_total{result,
source}`, `crisis_detected_total{detector}`, `fallback_total{reason}`,
`llm_tokens_total{model,type}` for cost, and `agent_requests_total` labelled by
outcome rather than status code. There are twelve metrics in all.

Traces carry the attributes that make them diagnostic: route, crisis,
confidence, model used, attempts, verdict and its reasons. One real capture
diagnosed a whole turn, showing empty retrieval, an empty draft, a rule
rejection, regeneration, failure and fixed text.

Logs exclude message content, replies, memory contents and the key. Nodes log
decisions, correlated to traces by `trace_id`, since a therapy transcript in a
log file is the privacy failure this project avoids elsewhere.

Alerts go to a local webhook receiver that logs them and serves recent ones at
`/alerts`, which is self-contained with no external account and no credential
in the repository. Telegram or SMTP is a receiver block and nothing else
changes. There are eight rules covering app down, verifier fail rate,
safe-fallback rate, router fallback, p95 latency, per-model error rate, memory
errors, and an informational rule for a sustained rise in crisis detections.

The path was verified rather than asserted: stopping the app produced pending,
then firing, then delivery through Alertmanager to the receiver, and restarting
produced the resolved notification.

One finding is worth volunteering. Phoenix returned 401 on every span push
while the application logged that tracing was enabled, because recent images
enable auth by default and OTLP export is fire-and-forget. There is now a
startup probe that logs the rejection at ERROR. Observability that reports
itself healthy while silently dropping everything is worse than none, because
it removes the suspicion that would prompt a check.

## What would you do next?

In priority order, all of it recorded in the docs rather than hidden.

1. Fix the retrieved-memory denial. Both generating models deny knowing a fact
   that is in their prompt when asked directly, while using it correctly when
   asked indirectly. Retrieval works; the model's use of it does not, in one
   specific shape. The isolating experiment was interrupted by the rate limit.
2. Finish the measurements. Seventeen eval cases and the whole remote model
   comparison need OpenRouter quota, and memory recall must be measured in the
   container where the vector index is active.
3. Add reranking and a relevance threshold on retrieval.
4. Store temporal metadata as a field rather than a prompt convention.
5. Add gVisor or a microVM before ever running model-generated code.
6. Set up CI running the tests and the offline evals on every change.
