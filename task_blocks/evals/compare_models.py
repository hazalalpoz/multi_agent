#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CATALOGUE_URL = "https://openrouter.ai/api/v1/models"


DEFAULT_REMOTE_CANDIDATES = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]


DEFAULT_LOCAL_CANDIDATES = ["qwen3:1.7b"]


GENERATION_CASE_IDS = [
    "route-emo-02",
    "route-log-02",
    "route-mixed-01",
    "crisis-01",
    "tool-01",
    "quality-03",
    "inject-02",
]


@dataclass
class RouterScore:
    model: str
    n: int = 0
    correct: int = 0
    parse_failures: int = 0
    crisis_tp: int = 0
    crisis_fn: int = 0
    crisis_fp: int = 0
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None

    @property
    def crisis_recall(self) -> float | None:
        total = self.crisis_tp + self.crisis_fn
        return self.crisis_tp / total if total else None


@dataclass
class GenerationScore:
    model: str
    replies: dict[str, str] = field(default_factory=dict)
    answered: int = 0
    rules_passed: int = 0
    n: int = 0
    latencies: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)


def next_run_id(prefix: str) -> str:
    existing = [
        int(match.group(1))
        for path in RESULTS_DIR.glob(f"{prefix}-*.json")
        if (match := re.match(rf"{prefix}-(\d+)\.json$", path.name))
    ]
    return f"{prefix}-{max(existing, default=0) + 1:03d}"


def load_cases() -> dict[str, dict[str, Any]]:
    return {
        case["id"]: case
        for case in (
            json.loads(line)
            for line in DATASET.read_text().splitlines()
            if line.strip()
        )
    }


def fetch_catalogue() -> dict[str, dict[str, Any]]:
    try:
        import httpx

        response = httpx.get(CATALOGUE_URL, timeout=30.0)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not fetch model catalogue ({exc})", file=sys.stderr)
        return {}

    catalogue: dict[str, dict[str, Any]] = {}
    for entry in response.json().get("data", []):
        params = entry.get("supported_parameters") or []
        pricing = entry.get("pricing") or {}
        catalogue[entry["id"]] = {
            "context_length": entry.get("context_length"),
            "tools": "tools" in params,
            "response_format": "response_format" in params,
            "prompt_usd_per_mtok": _per_mtok(pricing.get("prompt")),
            "completion_usd_per_mtok": _per_mtok(pricing.get("completion")),
        }
    return catalogue


def _per_mtok(value: Any) -> float | None:
    try:
        return round(float(value) * 1_000_000, 4)
    except (TypeError, ValueError):
        return None


def _build_model(model_id: str, *, local: bool, temperature: float):
    from task_blocks.config import get_settings

    settings = get_settings()
    if local:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_id,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            reasoning=False,
        )

    from langchain_openai import ChatOpenAI

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return ChatOpenAI(
        model=model_id,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=temperature,
        timeout=120,
        max_retries=1,
    )


def score_router(
    model_id: str, *, local: bool, cases: list[dict[str, Any]], sleep: float
) -> RouterScore:
    from langchain_core.messages import HumanMessage, SystemMessage

    from task_blocks.assets import load_prompt
    from task_blocks.graph import _parse_json_object
    from task_blocks.state import RouterDecision

    score = RouterScore(model=model_id)
    prompt = load_prompt("router")
    base = _build_model(model_id, local=local, temperature=0.0)
    structured = base.with_structured_output(
        RouterDecision, method="json_schema" if local else "json_mode"
    )

    for case in cases:
        expect = case.get("expect") or {}
        messages = [SystemMessage(content=prompt), HumanMessage(content=case["message"])]

        decision: RouterDecision | None = None
        started = time.perf_counter()
        try:
            result = structured.invoke(messages)
            decision = (
                result
                if isinstance(result, RouterDecision)
                else RouterDecision.model_validate(result)
            )
        except Exception:
            score.parse_failures += 1
            try:
                raw = base.invoke(messages)
                parsed = _parse_json_object(
                    raw.content if isinstance(raw.content, str) else str(raw.content)
                )
                if parsed:
                    decision = RouterDecision.model_validate(parsed)
            except Exception as exc:  # noqa: BLE001
                score.errors.append(f"{case['id']}: {type(exc).__name__}: {exc}")
        score.latencies.append(time.perf_counter() - started)

        if decision is None:
            score.n += 1
            if expect.get("crisis"):
                score.crisis_fn += 1
            continue

        score.n += 1
        if decision.route == expect.get("route"):
            score.correct += 1
        if expect.get("crisis") and decision.crisis:
            score.crisis_tp += 1
        elif expect.get("crisis") and not decision.crisis:
            score.crisis_fn += 1
        elif not expect.get("crisis") and decision.crisis:
            score.crisis_fp += 1

        if sleep and not local:
            time.sleep(sleep)

    return score


def score_generation(
    model_id: str,
    *,
    local: bool,
    cases: list[dict[str, Any]],
    sleep: float,
) -> GenerationScore:
    from langchain_core.messages import HumanMessage, SystemMessage

    from task_blocks.assets import build_system_prompt, get_skill
    from task_blocks.observability import token_usage
    from task_blocks.verification import check_reply

    score = GenerationScore(model=model_id)
    model = _build_model(model_id, local=local, temperature=0.4)

    for case in cases:
        expect = case.get("expect") or {}
        route = expect.get("route") or "logical"
        crisis = bool(expect.get("crisis"))
        role = "therapist" if route == "emotional" else "logical"

        extra: list[tuple[str, str]] = []
        if crisis:
            skill = get_skill("crisis-response")
            if skill is not None:
                extra.append(("Active skill: crisis-response", skill.body))


        system = build_system_prompt(role, extra_sections=extra)
        score.n += 1
        started = time.perf_counter()
        try:
            reply = model.invoke(
                [SystemMessage(content=system), HumanMessage(content=case["message"])]
            )
        except Exception as exc:  # noqa: BLE001
            score.errors.append(f"{case['id']}: {type(exc).__name__}: {exc}")
            score.latencies.append(time.perf_counter() - started)
            continue

        score.latencies.append(time.perf_counter() - started)
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        score.replies[case["id"]] = text
        usage = token_usage(reply)
        score.prompt_tokens += usage.get("prompt_tokens", 0)
        score.completion_tokens += usage.get("completion_tokens", 0)

        if text.strip():
            score.answered += 1
        if not check_reply(text, route=route, crisis=crisis):
            score.rules_passed += 1

        if sleep and not local:
            time.sleep(sleep)

    return score


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _p50(samples: list[float]) -> str:
    return f"{statistics.median(samples):.1f}s" if samples else "n/a"


def write_report(
    path: Path,
    router_scores: list[RouterScore],
    generation_scores: list[GenerationScore],
    catalogue: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    cases: dict[str, dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# LLM selection\n")
    lines.append(f"- **Run:** {meta['run_id']}")
    lines.append(f"- **Router cases:** {meta['router_case_count']}")
    lines.append(f"- **Generation cases:** {meta['generation_case_count']}")
    lines.append(f"- **Duration:** {meta['duration_s']:.1f}s\n")
    if meta.get("note"):
        lines.append(f"> {meta['note']}\n")

    lines.append("## Selection criteria\n")
    lines.append(
        "Candidates are judged on: accuracy on this system's own cases; "
        "crisis recall, which has a hard 100% target; structured-output "
        "reliability, since the router and verifier depend on schema-valid "
        "JSON; tool-calling support; latency; context window; cost and "
        "rate limits; licence and open weights, which the assignment "
        "requires; and privacy, meaning whether the endpoint logs prompts.\n"
    )

    if catalogue:
        lines.append("## Capabilities, from the provider catalogue\n")
        lines.append(
            "| Model | Context | Tools | JSON schema | $/Mtok in | $/Mtok out |"
        )
        lines.append("|---|---|---|---|---|---|")
        for model_id in meta["all_models"]:
            facts = catalogue.get(model_id)
            if not facts:
                lines.append(f"| `{model_id}` | local | yes | yes | 0 | 0 |")
                continue
            lines.append(
                f"| `{model_id}` | {facts['context_length'] or 'n/a'} "
                f"| {'yes' if facts['tools'] else 'no'} "
                f"| {'yes' if facts['response_format'] else 'no'} "
                f"| {facts['prompt_usd_per_mtok']} "
                f"| {facts['completion_usd_per_mtok']} |"
            )
        lines.append("")

    if router_scores:
        lines.append("## Router role\n")
        lines.append(
            "| Model | Accuracy | Crisis recall | Crisis FP | JSON failures | p50 latency |"
        )
        lines.append("|---|---|---|---|---|---|")
        for score in router_scores:
            lines.append(
                f"| `{score.model}` | {_pct(score.accuracy)} "
                f"| {_pct(score.crisis_recall)} | {score.crisis_fp} "
                f"| {score.parse_failures}/{score.n} | {_p50(score.latencies)} |"
            )
        lines.append("")
        for score in router_scores:
            if score.errors:
                lines.append(f"Errors for `{score.model}`:\n")
                for error in score.errors[:5]:
                    lines.append(f"- `{error}`")
                lines.append("")

    if generation_scores:
        lines.append("## Generation roles\n")
        lines.append(
            "| Model | Answered | Verifier rules passed | p50 latency | Prompt tok | Completion tok |"
        )
        lines.append("|---|---|---|---|---|---|")
        for score in generation_scores:
            lines.append(
                f"| `{score.model}` | {score.answered}/{score.n} "
                f"| {score.rules_passed}/{score.n} | {_p50(score.latencies)} "
                f"| {score.prompt_tokens or 'n/a'} "
                f"| {score.completion_tokens or 'n/a'} |"
            )
        lines.append("")
        for score in generation_scores:
            if score.errors:
                lines.append(f"Errors for `{score.model}`:\n")
                for error in score.errors[:5]:
                    lines.append(f"- `{error}`")
                lines.append("")

        lines.append("## Replies, side by side\n")
        lines.append(
            "Quality is not scored automatically — no judge model is used. "
            "These are here to be read.\n"
        )
        for case_id in meta["generation_case_ids"]:
            case = cases.get(case_id) or {}
            lines.append(f"### `{case_id}`\n")
            lines.append(f"**Message:** {case.get('message', '')}\n")
            for score in generation_scores:
                reply = score.replies.get(case_id)
                lines.append(f"**`{score.model}`**\n")
                if reply:
                    lines.append("> " + reply.strip().replace("\n", "\n> ") + "\n")
                else:
                    lines.append("> (no reply — see errors above)\n")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare candidate LLMs.")
    parser.add_argument(
        "--router-only",
        action="store_true",
        help="Score only the router role on local candidates — free and offline.",
    )
    parser.add_argument(
        "--models", action="append", help="Remote candidate (repeatable)."
    )
    parser.add_argument(
        "--local-models", action="append", help="Local Ollama candidate (repeatable)."
    )
    parser.add_argument("--limit", type=int, help="Maximum router cases per model.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=float(os.getenv("EVAL_SLEEP_SECONDS", "1.5")),
        help="Pause between remote calls, to respect free-tier limits.",
    )
    parser.add_argument("--note", default="", help="Note recorded in the report.")
    args = parser.parse_args()

    from task_blocks.logging_config import configure_logging

    configure_logging(os.getenv("LOG_LEVEL", "WARNING"))

    cases = load_cases()
    router_cases = [
        case
        for case in cases.values()
        if case["category"] in ("routing", "ambiguous", "figure_of_speech", "crisis")
        and (case.get("expect") or {}).get("route")
    ]
    if args.limit:
        router_cases = router_cases[: args.limit]

    local_candidates = args.local_models or DEFAULT_LOCAL_CANDIDATES
    remote_candidates = [] if args.router_only else (args.models or DEFAULT_REMOTE_CANDIDATES)

    run_id = next_run_id("models")
    started = time.perf_counter()

    router_scores: list[RouterScore] = []
    for model_id in local_candidates:
        print(f"router: {model_id} ({len(router_cases)} cases) ... ", end="", flush=True)
        score = score_router(model_id, local=True, cases=router_cases, sleep=0.0)
        router_scores.append(score)
        print(f"acc {_pct(score.accuracy)}, crisis recall {_pct(score.crisis_recall)}")

    for model_id in remote_candidates:
        print(f"router: {model_id} ({len(router_cases)} cases) ... ", end="", flush=True)
        score = score_router(
            model_id, local=False, cases=router_cases, sleep=args.sleep
        )
        router_scores.append(score)
        print(f"acc {_pct(score.accuracy)}, crisis recall {_pct(score.crisis_recall)}")

    generation_scores: list[GenerationScore] = []
    generation_cases = [cases[cid] for cid in GENERATION_CASE_IDS if cid in cases]
    if not args.router_only:
        for model_id in remote_candidates + local_candidates:
            is_local = model_id in local_candidates
            print(
                f"generation: {model_id} ({len(generation_cases)} cases) ... ",
                end="",
                flush=True,
            )
            score = score_generation(
                model_id,
                local=is_local,
                cases=generation_cases,
                sleep=args.sleep,
            )
            generation_scores.append(score)
            print(f"answered {score.answered}/{score.n}, rules {score.rules_passed}/{score.n}")

    catalogue = fetch_catalogue()
    duration = time.perf_counter() - started
    meta = {
        "run_id": run_id,
        "duration_s": duration,
        "router_case_count": len(router_cases),
        "generation_case_count": len(generation_cases) if not args.router_only else 0,
        "generation_case_ids": [c["id"] for c in generation_cases],
        "all_models": local_candidates + remote_candidates,
        "note": args.note
        or ("router-only run: no remote models called" if args.router_only else ""),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{run_id}.json"
    md_path = RESULTS_DIR / f"{run_id}.md"

    json_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "catalogue": catalogue,
                "router": [
                    {
                        "model": s.model,
                        "n": s.n,
                        "accuracy": s.accuracy,
                        "crisis_recall": s.crisis_recall,
                        "crisis_fp": s.crisis_fp,
                        "parse_failures": s.parse_failures,
                        "p50_latency_s": (
                            round(statistics.median(s.latencies), 3)
                            if s.latencies
                            else None
                        ),
                        "errors": s.errors,
                    }
                    for s in router_scores
                ],
                "generation": [
                    {
                        "model": s.model,
                        "n": s.n,
                        "answered": s.answered,
                        "rules_passed": s.rules_passed,
                        "prompt_tokens": s.prompt_tokens,
                        "completion_tokens": s.completion_tokens,
                        "p50_latency_s": (
                            round(statistics.median(s.latencies), 3)
                            if s.latencies
                            else None
                        ),
                        "replies": s.replies,
                        "errors": s.errors,
                    }
                    for s in generation_scores
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_report(md_path, router_scores, generation_scores, catalogue, meta, cases)

    print(f"\nwrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
