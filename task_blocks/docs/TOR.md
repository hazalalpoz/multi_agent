# Technical specification

## 1. Purpose

A conversational assistant that handles two kinds of turn well instead of one
kind badly.

People bring both sorts of thing to the same conversation. A message about a
cancelled flight may be mostly logistics, or mostly about what the trip was
for. A single prompt tuned for warmth gives vague answers to factual questions;
one tuned for precision answers a distressed person correctly and coldly. The
system therefore classifies each turn, dispatches it to a specialist, and
checks the result before anyone sees it.

A secondary purpose is to demonstrate a multi-agent architecture in which cost,
safety and observability are design constraints rather than later additions.

## 2. Scope

In scope:

- Emotional-support conversation: reflective listening, validation, open
  questions. This is not therapy.
- Factual and analytical answers, including tool use for anything
  time-dependent.
- Detection of crisis signals, and a response that always reaches real help.
- Memory within a conversation, and durable facts across conversations.
- Deployment in an isolated environment with metrics, logs, traces and alerts.
- Evaluation of routing, safety, verification and end-to-end behaviour.

Out of scope:

- Clinical assessment, diagnosis, treatment or crisis intervention.
- Authentication, multi-tenancy, consent management and data-retention policy.
  `user_id` is trusted as supplied, which is acceptable for coursework and
  would not be in production.
- Voice, images and non-English conversation.
- Human escalation or handoff. The system can point to help but cannot summon
  it.
- Horizontal scaling. The deployment is a single node using SQLite.

## 3. Users

| User | Needs |
|---|---|
| Someone upset | To be understood before being advised |
| Someone with a question | The answer first, honestly bounded |
| Someone at risk | To be taken seriously, and given help they can reach |
| Operator | To see what the system did and why, and to be told when it degrades |
| Assessor | To see the architecture, the decisions, the alternatives and real measurements |

## 4. Functional requirements

| ID | Requirement |
|---|---|
| F1 | Classify each turn as `emotional` or `logical` and dispatch accordingly |
| F2 | Detect crisis signals independently of that classification |
| F3 | Answer emotional turns with reflective listening rather than advice lists |
| F4 | Answer factual turns answer-first, with an honest "I don't know" where applicable |
| F5 | Call tools when the answer depends on information the model cannot have |
| F6 | Validate every reply before sending it, and regenerate a rejected one once |
| F7 | Send fixed, safe text if validation fails twice |
| F8 | Persist each conversation and resume it by `thread_id` |
| F9 | Store durable facts per `user_id` and retrieve them semantically |
| F10 | Never store anything from a crisis turn |
| F11 | Expose a CLI and an HTTP API, both reporting how a reply was produced |
| F12 | Load all prompts, skills and identity from Markdown at runtime |
| F13 | Route between models by role, keeping high-volume calls local |
| F14 | Fall back to another model on transport failure or rate limiting |

## 5. Non-functional requirements

| ID | Requirement | Status |
|---|---|---|
| N1 | Crisis recall of 100% on the eval set | Met: 7/7 measured |
| N2 | Routing accuracy of at least 85% | Met: 93.9% measured (n=33) |
| N3 | Verifier catches every seeded bad reply with no false rejections | Met: precision 100%, recall 100% (n=11) |
| N4 | Router adds under 2 seconds to a turn | Partly met: p50 1.27s, but p95 2.63s and a 5.77s worst case on the first call after the model loads |
| N5 | End-to-end p95 latency | Measured at 10 to 40 seconds per turn on free-tier remote models with CPU-only local inference. No target set, as it is provider-bound |
| N6 | Classification and verification cost nothing per request | Met: both run locally |
| N7 | Run as non-root, read-only filesystem, no capabilities | Met: verified |
| N8 | The inference engine has no internet access | Met: verified, zero default routes |
| N9 | No message content or secrets in logs at INFO | Met by construction |
| N10 | Metrics, logs, traces and alerting all working | Met: alert path verified firing and resolving |
| N11 | A missing observability backend must not cause an outage | Met: tracing degrades to a warning |
| N12 | Reproducible install and report build | Met: `uv.lock`, images pinned by digest |

## 6. Interfaces

CLI: `uv run task-blocks`, with `--user`, `--thread` and `--log-level`, plus
in-session `/info`, `/models` and `/memories`.

HTTP: `POST /chat` with `{user_id, thread_id?, message}`, returning the reply
plus `route`, `crisis`, `confidence`, `model_used`, `attempts`,
`memories_used`, `verdict`, `trace_id` and `latency_ms`. Also `GET /health` and
`GET /metrics`.

Both return how the answer was produced as well as the answer itself. For an
agentic system that is the difference between a usable interface and a black
box.

## 7. Constraints

- Open-weights models only, run locally where practical.
- Free-tier remote inference, so rate limits are a design constraint rather
  than an edge case. The daily cap was exhausted during development, which is
  what the fallback chain exists for.
- Single host with CPU-only local inference. There is no NVIDIA GPU, and Docker
  on macOS has no Metal access.
- Python 3.13 or later, with `uv` and LangGraph.
- The host's CPython cannot load SQLite extensions, so semantic memory search
  works only inside the container. Measurement has to account for this.

## 8. Risks

| Risk | Mitigation |
|---|---|
| A crisis signal is missed | Two independent detectors, keyword net as a floor under the model, a deterministic verifier rule, and 100% recall as a release gate |
| A reply gives harmful information | The crisis skill forbids it, the verifier checks, and fixed safe text is the floor |
| The model claims clinical authority | Forbidden in `SOUL.md`, checked by rule, covered by a test |
| Memories leak between users | Namespace scoping per `user_id`, verified |
| Sensitive data stored permanently | Save policy in the skill; crisis turns have no memory tools at all |
| Prompt injection | Instruction-source boundary in every prompt, and injection cases in the eval set |
| Free-tier rate limiting | Transport fallback, then regeneration, then fixed text |
| Model drift | Verifier pass rate and fallback rate alerted on, with periodic eval runs |
| Observability failing silently | Startup probe; a 401 from the collector is logged at ERROR |

## 9. Acceptance

The system is acceptable when every functional requirement is implemented in
the graph, crisis recall is 100% on the eval set, the verifier makes no false
rejections on the seeded set, the stack comes up from `docker compose up` with
`/health` reporting `ok`, the inference engine provably has no egress, an alert
fires and resolves through the full path, and the report cites only numbers
produced by scripts in this repository.
