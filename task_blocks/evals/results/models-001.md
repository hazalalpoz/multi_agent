# LLM selection

- **Run:** models-001
- **Router cases:** 33
- **Generation cases:** 0
- **Duration:** 43.4s

> Router role only, local candidate. Remote candidates not run: OpenRouter daily free-tier quota exhausted. Model-only scoring, without the deterministic keyword net the graph applies on top.

## Selection criteria

Candidates are judged on: accuracy on this system's own cases; crisis recall, which has a hard 100% target; structured-output reliability, since the router and verifier depend on schema-valid JSON; tool-calling support; latency; context window; cost and rate limits; licence and open weights, which the assignment requires; and privacy, meaning whether the endpoint logs prompts.

## Capabilities, from the provider catalogue

| Model | Context | Tools | JSON schema | $/Mtok in | $/Mtok out |
|---|---|---|---|---|---|
| `qwen3:1.7b` | local | yes | yes | 0 | 0 |

## Router role

| Model | Accuracy | Crisis recall | Crisis FP | JSON failures | p50 latency |
|---|---|---|---|---|---|
| `qwen3:1.7b` | 91% | 86% | 1 | 0/33 | 1.3s |
