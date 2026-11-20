# Agents

Every agent in the system, with its model, inputs, outputs and handoffs. The
graph that wires them together is `src/task_blocks/graph.py`; the diagrams in
`docs/ARCHITECTURE.md` are drawn from the same node names used here.

```
START → load_memory → router ──emotional──→ therapist ─┐
                         │                             ├─⇄ tools
                         └──logical────→  logical ─────┘
                                              ↓
                                          verifier ──pass──→ save_turn → END
                                              │ fail (attempt 1)
                                              └──→ regenerate → verifier
                                                   fail again → safe fallback → END
```

Nodes that are not agents (`load_memory`, `save_turn`) are listed last.

---

## router

Classifies the incoming turn and detects crisis signals. It is the only agent
that sees nothing but the latest user message plus retrieved memories.

| | |
|---|---|
| **Model** | `qwen3:1.7b` (local, Ollama), thinking disabled, temperature 0 |
| **Prompt** | `agent/prompts/router.md` |
| **Tools** | none |
| **Input** | latest user message, retrieved `memories` |
| **Output** | `route` (`emotional` \| `logical`), `crisis` (bool), `confidence` (0–1) |
| **Handoff** | `therapist` when `route == "emotional"`, else `logical` |

Structured JSON through Ollama's schema-constrained `format`. If the output
cannot be parsed, the node falls back to `route="logical"` and sets `crisis`
from a deterministic keyword check, then logs the fallback with the reason. A
crisis signal forces `route="emotional"` regardless of what the model said.

Why a local 1.7B model: classification is the highest-volume LLM call in the
system and the cheapest to get right. Running it locally means routing costs
nothing per request and leaks nothing to a third party.

## therapist

Handles emotional turns: distress, relationships, loss, self-doubt, anything
where being understood matters more than being answered.

| | |
|---|---|
| **Model** | `google/gemma-4-31b-it:free` (OpenRouter), temperature 0.7 |
| **Prompt** | `agent/prompts/therapist.md` + `SOUL.md` + skill index |
| **Tools** | `search_memory`, `manage_memory`, `load_skill`, `get_datetime` |
| **Input** | trimmed conversation history, `memories`, `crisis`, `skill_hint` |
| **Output** | `draft`, `model_used` |
| **Handoff** | `verifier` |

A ReAct loop: it may call tools several times before answering. When `crisis` is
true the body of the crisis-response skill is preloaded into its prompt rather
than left for it to fetch, so the safety path never depends on the model
choosing to call a tool.

## logical

Handles factual, analytical and practical turns.

| | |
|---|---|
| **Model** | `google/gemma-4-26b-a4b-it:free` (OpenRouter), temperature 0.2 |
| **Prompt** | `agent/prompts/logical.md` + `SOUL.md` + skill index |
| **Tools** | `search_memory`, `manage_memory`, `load_skill`, `get_datetime` |
| **Input** | trimmed conversation history, `memories`, `skill_hint` |
| **Output** | `draft`, `model_used` |
| **Handoff** | `verifier` |

Also a ReAct loop. A sparse MoE with roughly 4B active parameters: markedly
faster and cheaper per token than the therapist model, which suits a role that
makes more tool calls per turn.

## verifier

The validation layer. Nothing reaches the user without passing through it.

| | |
|---|---|
| **Model** | `qwen3:1.7b` (local, Ollama), temperature 0 |
| **Prompt** | `agent/prompts/verifier.md` |
| **Tools** | none |
| **Input** | `draft`, latest user message, `route`, `crisis` |
| **Output** | `verdict` (`passed` + `reasons[]` + `source`) |
| **Handoff** | `save_turn` on pass; `regenerate` on fail when `attempts` allows; safe fallback reply otherwise |

Hybrid, in two stages:

1. **Deterministic rules**, in code, no model involved: empty or over-long
   reply, a leaked tool error, a leaked system prompt, a crisis turn with no
   help resources, an emotional turn answered as pure fact or a logical turn
   answered as pure sentiment.
2. **Model judgement** on what rules cannot see: does this answer the question,
   does it match the route's register, is it safe?

Deterministic failures override the model verdict in both directions — a rule
failure fails the draft even if the model passed it. The rules are the part we
can test and the part that cannot hallucinate, so they get the last word.

## regenerate

One retry, on a different model, with the verifier's objections made explicit.
This is the evaluator-optimizer pattern: the critique becomes input.

| | |
|---|---|
| **Model** | `nvidia/nemotron-3-ultra-550b-a55b:free` (OpenRouter) |
| **Prompt** | `agent/prompts/regenerate.md` |
| **Tools** | none |
| **Input** | original request, rejected `draft`, `verdict.reasons` |
| **Output** | new `draft`, `model_used`, `attempts += 1` |
| **Handoff** | `verifier` |

Capped at `MAX_REGENERATION_ATTEMPTS` (default 1). If the retry also fails, the
system sends a fixed, hand-written safe reply instead of a generated one — help
resources for a crisis turn, an honest "I can't answer this well" otherwise.
A fixed reply is the only output we can guarantee is safe, so it is what a
double failure gets.

Switching models rather than retrying the same one is deliberate: a model that
produced a bad draft at temperature is likely to produce a similar one again.

---

## Non-agent nodes

**load_memory** — semantic search of the long-term store, namespace
`("memories", user_id)`, using the latest user message as the query. Writes
`memories`. No model call; embeddings come from local `nomic-embed-text`.

**save_turn** — the checkpointer persists the thread under `thread_id`. Facts
reach long-term memory only when an agent writes them with `manage_memory`,
never automatically: what is worth remembering is a judgement, and silently
storing everything is both a privacy problem and a retrieval problem.
