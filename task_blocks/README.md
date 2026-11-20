# task-blocks

A multi-agent assistant built on LangGraph. It handles two kinds of
conversation, emotional support and factual questions. A router classifies each
turn and sends it to the matching specialist agent, a verification layer checks
every reply before it is sent, and long-term memory carries facts between
conversations.

Cost is controlled by routing: classification and verification run on a 1.7B
model locally, and only generation reaches a larger remote model.

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

Full documentation is in [`docs/`](docs): the specification
([TOR.md](docs/TOR.md)), the architecture with C4 and sequence diagrams
([ARCHITECTURE.md](docs/ARCHITECTURE.md)), the decisions and the alternatives
rejected ([DECISIONS.md](docs/DECISIONS.md)), plus
[MEMORY.md](docs/MEMORY.md), [EVALS.md](docs/EVALS.md),
[OBSERVABILITY.md](docs/OBSERVABILITY.md) and
[DEFENSE_QA.md](docs/DEFENSE_QA.md).

## The model routing table

| Role | Model | Engine |
|---|---|---|
| router, verifier | `qwen3:1.7b` | local Ollama |
| therapist | `google/gemma-4-31b-it:free` | OpenRouter |
| logical | `google/gemma-4-26b-a4b-it:free` | OpenRouter |
| fallback (429, error, verifier reject) | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter |
| embeddings | `nomic-embed-text` | local Ollama |

Every model is open-weights. All IDs live in
[`src/task_blocks/config.py`](src/task_blocks/config.py) and are overridable by
environment variable.

The engine follows the model ID: a name containing a slash goes to OpenRouter,
one without it goes to the local Ollama. So the whole system can run offline,
which is useful when the OpenRouter free tier's daily cap is spent:

```bash
THERAPIST_MODEL=qwen3:1.7b LOGICAL_MODEL=qwen3:1.7b FALLBACK_MODEL=qwen3:1.7b uv run task-blocks
```

Replies are weaker from a 1.7B model, but routing, verification, tools and
memory all behave normally.


## Requirements

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- Docker, for the deployed stack and for building the report
- An [OpenRouter](https://openrouter.ai/keys) API key for the generating models
- Ollama, for `qwen3:1.7b` and `nomic-embed-text`. The Compose stack runs its
  own; for host runs, point `OLLAMA_BASE_URL` at yours

## Setup

```bash
uv sync
```

```bash
cp .env.example .env
```

Then put your key in `.env` as `OPENROUTER_API_KEY`. `.env` is gitignored and
no value in it is ever logged or baked into an image. Every variable is listed
in [`.env.example`](.env.example) with its default.

## Running it

### Docker Compose (the real deployment)

```bash
docker compose up -d --build
```

Eight services start: `app`, `ollama`, a one-shot `ollama-init` that pulls the
local models, and the observability stack (`phoenix`, `prometheus`, `grafana`,
`alertmanager`, `alert-webhook`). The app waits for Ollama to be healthy, so
the first request never races a model download.

| Service | URL | What it is |
|---|---|---|
| app | <http://127.0.0.1:8000> | the API |
| phoenix | <http://127.0.0.1:6006> | traces |
| prometheus | <http://127.0.0.1:9090> | metrics and alert rules |
| grafana | <http://127.0.0.1:3000> | dashboard (`admin`/`admin` by default) |
| alertmanager | <http://127.0.0.1:9093> | alert routing |
| alert-webhook | <http://127.0.0.1:8081/alerts> | where alerts land |

Every UI binds to `127.0.0.1`. Ollama's port is deliberately not published, so
this stack cannot collide with an Ollama you already run.

```bash
docker compose down
```

Volumes survive, so conversations and memories outlive the containers.

The stack's Ollama is not reachable from the host, by design. To run the CLI
or the tests on the host against it, add the development override, which
publishes the port on `127.0.0.1` and is explained in
[`compose.dev.yaml`](compose.dev.yaml):

```bash
docker compose -f compose.yaml -f compose.dev.yaml up -d
```

Do not use that override for the deployment being assessed: it gives the
inference engine egress, which the production file removes.

### CLI

```bash
uv run task-blocks
```

Options: `--user` scopes long-term memory, `--thread` resumes a previous
conversation, `--log-level` overrides `LOG_LEVEL`. In-session commands:
`/help`, `/info` (route, model and verdict for the last turn), `/models`,
`/memories`, `/exit`.

### API

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H 'content-type: application/json' -d '{"user_id":"alice","message":"What is the capital of Portugal?"}'
```

`POST /chat` takes `{user_id, thread_id?, message}` and returns the reply plus
how it was produced: `route`, `crisis`, `confidence`, `model_used`,
`attempts`, `memories_used`, `verdict` and `trace_id`. Omit `thread_id` to
start a new conversation.

`GET /health` reports Ollama reachability, missing models, storage and whether
vector search is available. `GET /metrics` is the Prometheus endpoint.

## Tests and evals

```bash
uv run pytest -q
```

141 tests covering the deterministic parts: verifier rules, the crisis keyword
net, the router's failure path, asset loading and tool error handling. These use
no network and no models, and run in under a second.

```bash
uv run python evals/run_evals.py --offline
```

Runs the cases that need only the local models and writes
`evals/results/run-NNN.json` and `.md`, numbered sequentially. Drop `--offline` to run all 61
cases, which needs OpenRouter quota. `--api http://127.0.0.1:8000` runs them
against the container instead. That is the only way to measure memory recall
with the vector index active, since macOS Python builds cannot load the
`sqlite-vec` extension. Other flags: `--category`, `--limit`, `--sleep`.

```bash
uv run python evals/compare_models.py --router-only
```

Scores candidate models for the router role on this system's own cases. Without
`--router-only` it also compares the generating candidates, which needs quota.

Latest measured results and their caveats are in [EVALS.md](docs/EVALS.md).

## The report

```bash
make report
```

Builds `report/report.pdf` from `docs/`, rendering every Mermaid diagram to an
image. The toolchain (mermaid-cli, pandoc, typst) runs in containers pinned by
digest, so Docker is the only host requirement. `make help` lists the other
targets.

## Layout

```
agent/            prompts, skills and SOUL.md, loaded at runtime (never inlined in Python)
  prompts/        one file per agent role
  skills/         <name>/SKILL.md with YAML frontmatter
src/task_blocks/  config, graph, models, memory, tools, verification, observability, api, cli
ops/              Prometheus, Grafana, Alertmanager and the webhook receiver
evals/            dataset, harness, model comparison, results
tests/            pytest suite for the deterministic parts
docs/             specification, architecture, decisions, evals, observability, defence notes
scripts/          report build
```
