# Architecture

Four views: system context (C4 L1), containers (C4 L2), components inside the
app (C4 L3), and the sequence of one turn. Every node named here is an explicit
node of one `StateGraph` in
[`src/task_blocks/graph.py`](../src/task_blocks/graph.py), so the diagrams can
be checked against the code.

## Patterns

Router: classify once, then dispatch to a specialist. One agent with one prompt
covering both registers produces replies that are mediocre in both.

ReAct: each specialist loops between reasoning and tool calls until it has what
it needs.

Evaluator-optimizer (reflection): the verifier criticises the draft, and the
critique becomes input to a regeneration. The critic is a separate component
from the generator, and is partly deterministic, so it cannot approve its own
work.

Fallback chain at two levels: transport failures are handled inside the model
client, while quality failures are handled by the graph, since they need the
verifier's reasons. Both end at fixed text rather than an error.

Progressive disclosure: agents see skill names and descriptions, and bodies are
fetched on demand. The base prompt stays the same size as the skill library
grows.

## L1, system context

```mermaid
graph TB
    user["Person<br/><i>upset, curious, or at risk</i>"]
    operator["Operator<br/><i>runs and watches the system</i>"]

    system["<b>task-blocks</b><br/>Multi-agent assistant:<br/>routes between an emotional-support<br/>agent and a factual agent, verifies<br/>every reply, remembers what matters"]

    openrouter["OpenRouter<br/><i>hosted open-weights models</i><br/>generation only"]
    ollama["Ollama<br/><i>local inference</i><br/>classification, verification, embeddings"]
    helplines["Published help resources<br/><i>112, findahelpline.com</i><br/>named in replies, never called"]

    user -->|"messages, in natural language"| system
    system -->|"replies, plus how they were produced"| user
    operator -->|"reads dashboards and traces"| system
    system -->|"alerts when it degrades"| operator

    system -->|"HTTPS: generation"| openrouter
    system -->|"HTTP, internal network: routing,<br/>verification, embeddings"| ollama
    system -.->|"points the person to"| helplines

    classDef focus fill:#1f6feb,stroke:#0b4ea2,color:#fff
    classDef ext fill:#eef2f6,stroke:#8899aa,color:#1a1a1a
    classDef person fill:#ffffff,stroke:#444,color:#1a1a1a
    class system focus
    class openrouter,ollama,helplines ext
    class user,operator person
```

There are two inference providers, split by role rather than by availability.
Classification and verification run on every single request, so their marginal
cost dominates: locally it is zero, and the text never leaves the host, which
matters when the text may be someone describing a crisis. Generation is where
model quality shows, so it goes to a larger remote model. This split is the
cost-optimisation strategy.

## L2, containers

```mermaid
graph TB
    user["Person"]
    operator["Operator"]

    subgraph frontend["Network: frontend — has egress"]
        app["<b>app</b><br/>Python 3.13, FastAPI, LangGraph<br/>the graph, tools, memory<br/>non-root, read-only rootfs, no capabilities<br/>:8000"]
        phoenix["<b>phoenix</b><br/>Arize Phoenix<br/>trace store and UI<br/>:6006"]
        prometheus["<b>prometheus</b><br/>metrics, alert rules<br/>:9090"]
    end

    subgraph backend["Network: backend — internal, NO egress"]
        ollama["<b>ollama</b><br/>qwen3:1.7b, nomic-embed-text<br/>port not published"]
    end

    subgraph obs["Network: observability"]
        grafana["<b>grafana</b><br/>dashboard from the repo<br/>:3000"]
        alertmanager["<b>alertmanager</b><br/>routing, grouping, inhibition<br/>:9093"]
        webhook["<b>alert-webhook</b><br/>logs what it receives<br/>:8081"]
    end

    initjob["<b>ollama-init</b><br/>one-shot: downloads models<br/>into the volume, then exits"]

    appdata[("app-data<br/><i>checkpoints.sqlite</i><br/><i>store.sqlite</i>")]
    models[("ollama-models")]

    user -->|"HTTP /chat"| app
    operator -->|"browses"| grafana
    operator -->|"browses"| phoenix

    app -->|"OTLP spans"| phoenix
    prometheus -->|"scrapes /metrics"| app
    prometheus -->|"fires alerts"| alertmanager
    alertmanager -->|"webhook"| webhook
    grafana -->|"queries"| prometheus

    app -->|"HTTP, no egress path"| ollama
    app --> appdata
    ollama --> models
    initjob -->|"writes models"| models

    external["OpenRouter API"]
    app -->|"HTTPS"| external

    classDef svc fill:#1f6feb,stroke:#0b4ea2,color:#fff
    classDef obsvc fill:#6f42c1,stroke:#4c2889,color:#fff
    classDef store fill:#fff8e1,stroke:#b08900,color:#1a1a1a
    classDef ext fill:#eef2f6,stroke:#8899aa,color:#1a1a1a
    class app,ollama svc
    class phoenix,prometheus,grafana,alertmanager,webhook obsvc
    class appdata,models store
    class external,initjob ext
```

The networks are split this way because `ollama` holds the local models and
needs no internet once they are downloaded. It therefore sits on a network with
no route out, so that if a prompt injection ever persuaded a tool to talk to
it, there would be nowhere for data to go. The `app` service needs egress
because generation is remote. The monitoring stack never joins `backend`, since
nothing in it has reason to talk to the inference engine.

The `ollama-init` service exists because `ollama pull` is a client call: the
server performs the download. Asking the no-egress server to pull fails by
design, so the download happens in a short-lived container that does have
egress and writes into the shared volume. The serving instance then starts from
that volume with no route out.

## L3, components inside the app

```mermaid
graph TB
    subgraph interfaces["Interfaces"]
        cli["cli.py<br/><i>interactive session</i>"]
        api["api.py<br/><i>/chat, /health, /metrics</i>"]
    end

    subgraph graphmod["graph.py — one StateGraph"]
        load["<b>load_memory</b><br/>semantic search for this user"]
        router["<b>router</b><br/>qwen3:1.7b, JSON schema<br/>route + crisis + confidence"]
        therapist["<b>therapist</b><br/>ReAct, gemma-4-31b"]
        logical["<b>logical</b><br/>ReAct, gemma-4-26b-a4b"]
        verifier["<b>verifier</b><br/>rules, then qwen3:1.7b"]
        regen["<b>regenerate</b><br/>nemotron + verifier reasons"]
        safe["<b>safe_fallback</b><br/>fixed text"]
        save["<b>save_turn</b>"]
    end

    subgraph support["Support modules"]
        config["config.py<br/><i>the routing table</i>"]
        models["models.py<br/><i>clients, transport fallbacks</i>"]
        assets["assets.py<br/><i>prompts, skills, SOUL</i>"]
        verification["verification.py<br/><i>deterministic rules,<br/>crisis + method nets</i>"]
        tools["tools.py<br/><i>load_skill, get_datetime,<br/>search_memory, manage_memory</i>"]
        memory["memory.py<br/><i>checkpointer + vector store</i>"]
        obs["observability.py<br/><i>metrics, spans</i>"]
    end

    cli --> load
    api --> load
    load --> router
    router -->|"emotional"| therapist
    router -->|"logical"| logical
    therapist --> verifier
    logical --> verifier
    verifier -->|"pass"| save
    verifier -->|"fail, attempt 1"| regen
    verifier -->|"fail again"| safe
    regen --> verifier

    therapist <-->|"tool calls"| tools
    logical <-->|"tool calls"| tools
    load --> memory
    tools --> memory
    router --> verification
    verifier --> verification
    therapist --> assets
    logical --> assets
    router --> models
    models --> config

    classDef node fill:#1f6feb,stroke:#0b4ea2,color:#fff
    classDef guard fill:#d73a49,stroke:#99202c,color:#fff
    classDef sup fill:#f6f8fa,stroke:#8899aa,color:#1a1a1a
    class load,router,therapist,logical,regen,save node
    class verifier,safe guard
    class config,models,assets,verification,tools,memory,obs,cli,api sup
```

The verification layer is drawn in red because the architecture is organised
around it. Everything else produces text; it decides whether text may be sent.

### State

One `TypedDict`, in [`state.py`](../src/task_blocks/state.py). The `messages`
field uses the `add_messages` reducer, and everything else is replaced by
whichever node writes it.

| Field | Written by | Purpose |
|---|---|---|
| `messages` | `save_turn`, `safe_fallback`, interfaces | the conversation |
| `route` | `router` | which specialist; also the verifier's register check |
| `crisis` | `router` | forces the emotional path, preloads the skill, withholds memory tools, enables a verifier rule |
| `confidence` | `router` | ambiguity signal, recorded as a histogram |
| `skill_hint` | `router` | which skill to preload |
| `draft` | specialists, `regenerate` | the candidate reply, not yet sent |
| `verdict` | `verifier` | pass or fail, reasons, and which stage decided |
| `attempts` | `regenerate` | terminates the loop |
| `model_used` | specialists, `regenerate`, `safe_fallback` | which model answered |
| `user_id`, `thread_id` | interfaces | memory scoping |
| `memories` | `load_memory` | retrieved long-term facts |

The `draft` field is separate from `messages` so that a reply becomes part of
the conversation only once it has passed. A rejected draft leaves no trace in
the thread.

## Sequence of one turn

A crisis turn, since it exercises every guard. This is the behaviour observed
in testing, where an explicit request to remember something was correctly not
stored.

```mermaid
sequenceDiagram
    autonumber
    actor P as Person
    participant API as api.py
    participant G as Graph
    participant ST as Store (SQLite + vectors)
    participant Q as Ollama qwen3:1.7b
    participant R as Rules (code)
    participant M as OpenRouter gemma-4-31b
    participant CP as Checkpointer

    P->>API: POST /chat {user_id, message}
    API->>G: run_turn()

    G->>ST: search("memories", user_id)
    ST-->>G: retrieved facts

    G->>Q: classify (JSON schema, thinking off)
    Q-->>G: {route, crisis, confidence}
    G->>R: keyword + method nets
    R-->>G: crisis signal
    Note over G,R: Either detector sets crisis.<br/>The nets are a floor under the model:<br/>one eval case was caught only here.

    G->>G: crisis → emotional route,<br/>preload crisis skill,<br/>withhold memory tools

    G->>M: generate (SOUL + prompt + skill + memories)
    M-->>G: draft

    G->>R: deterministic checks
    R-->>G: crisis reply must name help resources
    G->>Q: judge (answers? register? safe?)
    Q-->>G: {pass, reasons}
    Note over G,R: A rule failure overrides the model.<br/>Rules are testable and cannot hallucinate.

    alt verdict passes
        G->>CP: persist thread
        G-->>API: reply + route + model + verdict
    else rejected, first attempt
        G->>M: regenerate on fallback model,<br/>with the reasons
        M-->>G: new draft
        G->>R: re-verify
        Note over G: Rejected twice → fixed safe reply,<br/>which includes help resources
    end

    API-->>P: reply, with trace_id
    Note over P,CP: Nothing was written to long-term memory:<br/>the tools were not available on this turn.
```

## Reasoning behind the boundaries

The router is its own node rather than a function inside the specialist because
its decision is state the rest of the graph reads. The verifier checks register
against `route`, metrics are labelled by it, and the crisis flag changes which
tools exist. Hiding it inside a specialist would mean the specialist decides
whether it should have been called.

The verifier is a separate component rather than a longer system prompt because
a model asked to self-check in the same call has already committed to its
answer, and its self-assessment comes from the same process that produced the
mistake. A separate component, on a different model, with deterministic rules
that override it, can fail its input. In measurement it rejected 7 of 7
genuinely bad drafts and 0 of 4 acceptable ones.

Deterministic rules run first because they can be unit tested, cannot
hallucinate, and cost nothing. The model handles what rules cannot see, namely
whether the reply actually answers the question. Testing found a rule that had
never fired because of a regex bug, which supports this split: the untestable
half would have hidden it.

Regeneration switches model because a model that produced a bad draft at
temperature will likely produce a similar one again. Switching costs nothing
extra, since the fallback model is already configured for transport failures.

One retry is allowed because the loop must terminate, and a second failure is
evidence that the problem is not a sampling accident. Fixed text is the only
output that can be guaranteed safe.

Memory retrieval happens before routing because retrieval is cheap and local,
and a turn that turns out to be a crisis must not depend on a tool call for its
context. It also means the router sees the same recollections the specialists
do.
