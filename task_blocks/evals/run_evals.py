#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


OFFLINE_CATEGORIES = {
    "routing",
    "ambiguous",
    "figure_of_speech",
    "verifier_seeded",
    "crisis",
}


ROUTE_ONLY_CATEGORIES = {"routing", "ambiguous", "figure_of_speech"}


@dataclass
class CaseResult:
    case_id: str
    category: str
    passed: bool
    checks: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    observed: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None
    skipped: str | None = None


def next_run_id(prefix: str) -> str:
    existing = [
        int(match.group(1))
        for path in RESULTS_DIR.glob(f"{prefix}-*.json")
        if (match := re.match(rf"{prefix}-(\d+)\.json$", path.name))
    ]
    return f"{prefix}-{max(existing, default=0) + 1:03d}"


def load_cases(
    path: Path, *, categories: set[str] | None, limit: int | None
) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if categories:
        cases = [case for case in cases if case["category"] in categories]
    if limit:
        cases = cases[:limit]
    return cases


def contains_any(text: str, needles: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(needle.lower() in lowered for needle in needles if needle)


def excludes_all(text: str, needles: list[str]) -> list[str]:
    lowered = (text or "").lower()
    return [needle for needle in needles if needle and needle.lower() in lowered]


def tools_used(state: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in state.get("messages") or []:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else None
            if name:
                names.append(name)
    return names


class GraphRunner:
    def __init__(self) -> None:
        from task_blocks.graph import build_graph

        self.graph = build_graph()

    def turn(self, message: str, *, user_id: str, thread_id: str) -> dict[str, Any]:
        from task_blocks.graph import reply_text, run_turn

        state = run_turn(self.graph, message, user_id=user_id, thread_id=thread_id)
        return {
            "reply": reply_text(state),
            "route": state.get("route"),
            "crisis": bool(state.get("crisis")),
            "confidence": state.get("confidence"),
            "model_used": state.get("model_used"),
            "attempts": int(state.get("attempts", 0)),
            "verdict": state.get("verdict") or {},
            "memories_used": len(state.get("memories") or []),
            "tools": tools_used(state),
        }

    def route_only(self, message: str) -> dict[str, Any]:
        from task_blocks.graph import router

        result = router({"messages": [{"role": "user", "content": message}]})
        return {
            "route": result.get("route"),
            "crisis": bool(result.get("crisis")),
            "confidence": result.get("confidence"),
        }

    def verify_seeded(
        self, draft: str, *, route: str, crisis: bool
    ) -> dict[str, Any]:
        from task_blocks.graph import verifier

        state = {
            "messages": [{"role": "user", "content": "(seeded case)"}],
            "draft": draft,
            "route": route,
            "crisis": crisis,
            "attempts": 0,
        }
        return verifier(state)["verdict"]


class ApiRunner:
    def __init__(self, base_url: str) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=300.0)

    def turn(self, message: str, *, user_id: str, thread_id: str) -> dict[str, Any]:
        response = self.client.post(
            f"{self.base_url}/chat",
            json={"user_id": user_id, "thread_id": thread_id, "message": message},
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "reply": payload.get("reply", ""),
            "route": payload.get("route"),
            "crisis": bool(payload.get("crisis")),
            "confidence": payload.get("confidence"),
            "model_used": payload.get("model_used"),
            "attempts": int(payload.get("attempts", 0)),
            "verdict": payload.get("verdict") or {},
            "memories_used": int(payload.get("memories_used", 0)),


            "tools": None,
        }

    def route_only(self, message: str) -> dict[str, Any]:
        result = self.turn(message, user_id="eval-route", thread_id=f"r-{uuid.uuid4().hex[:8]}")
        return {
            "route": result["route"],
            "crisis": result["crisis"],
            "confidence": result["confidence"],
        }

    def verify_seeded(self, draft: str, *, route: str, crisis: bool) -> dict[str, Any]:
        raise RuntimeError("seeded verifier cases require --no-api (in-process)")


def run_routing_case(runner: Any, case: dict[str, Any]) -> CaseResult:
    expect = case.get("expect") or {}
    started = time.perf_counter()
    observed = runner.route_only(case["message"])
    latency = time.perf_counter() - started

    failures: list[str] = []
    if "route" in expect and observed["route"] != expect["route"]:
        failures.append(f"route {observed['route']!r} != {expect['route']!r}")
    if "crisis" in expect and observed["crisis"] != expect["crisis"]:
        failures.append(f"crisis {observed['crisis']} != {expect['crisis']}")

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        passed=not failures,
        failures=failures,
        observed=observed,
        latency_s=latency,
    )


def run_seeded_verifier_case(runner: Any, case: dict[str, Any]) -> CaseResult:
    expect = case.get("expect") or {}
    started = time.perf_counter()
    verdict = runner.verify_seeded(
        case.get("seeded_draft", ""),
        route=case.get("route", "logical"),
        crisis=bool(case.get("crisis")),
    )
    latency = time.perf_counter() - started

    failures: list[str] = []
    if "verdict_passed" in expect and bool(verdict.get("passed")) != expect["verdict_passed"]:
        failures.append(
            f"verdict passed={verdict.get('passed')} != {expect['verdict_passed']}"
        )
    wanted_reason = expect.get("reason_contains")
    if wanted_reason:
        joined = " ".join(verdict.get("reasons") or []).lower()
        if wanted_reason.lower() not in joined:
            failures.append(f"no reason mentioning {wanted_reason!r}; got {joined!r}")

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        passed=not failures,
        failures=failures,
        observed={"verdict": verdict},
        latency_s=latency,
    )


def _check_turn(expect: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    if "route" in expect and observed.get("route") != expect["route"]:
        failures.append(f"route {observed.get('route')!r} != {expect['route']!r}")
    if "crisis" in expect and observed.get("crisis") != expect["crisis"]:
        failures.append(f"crisis {observed.get('crisis')} != {expect['crisis']}")

    needles = expect.get("reply_contains_any")
    if needles and not contains_any(observed.get("reply", ""), needles):
        failures.append(f"reply contains none of {needles}")

    forbidden = expect.get("reply_excludes")
    if forbidden:
        found = excludes_all(observed.get("reply", ""), forbidden)
        if found:
            failures.append(f"reply contains forbidden text {found}")

    wanted_tools = expect.get("tools_used_any")
    if wanted_tools:
        used = observed.get("tools")
        if used is None:
            failures.append("tool use not observable in this run mode")
        elif not set(wanted_tools) & set(used):
            failures.append(f"none of tools {wanted_tools} called (called {used})")

    if "memory_writes" in expect:
        used = observed.get("tools")
        if used is not None:
            writes = sum(1 for name in used if name == "manage_memory")
            if writes != expect["memory_writes"]:
                failures.append(
                    f"{writes} memory writes, expected {expect['memory_writes']}"
                )

    return failures


def run_turn_case(runner: Any, case: dict[str, Any], *, user_id: str) -> CaseResult:
    turns = case.get("turns") or [
        {"message": case["message"], "expect": case.get("expect") or {}}
    ]
    case_user = case.get("user_id") or user_id
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"

    failures: list[str] = []
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, turn in enumerate(turns):
        if turn.get("new_thread"):
            thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
        observed = runner.turn(
            turn["message"], user_id=case_user, thread_id=thread_id
        )
        observations.append(observed)
        turn_expect = turn.get("expect") or (
            case.get("expect") if len(turns) == 1 else {}
        ) or {}
        for failure in _check_turn(turn_expect, observed):
            failures.append(f"turn {index + 1}: {failure}")

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        passed=not failures,
        failures=failures,
        observed={"turns": observations},
        latency_s=time.perf_counter() - started,
    )


def routing_metrics(results: list[CaseResult], cases_by_id: dict[str, dict]) -> dict:
    labelled = [
        r
        for r in results
        if r.observed.get("route") is not None
        and (cases_by_id[r.case_id].get("expect") or {}).get("route")
    ]
    if not labelled:
        return {}

    classes = ("emotional", "logical")
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    correct = 0
    for result in labelled:
        expected = cases_by_id[result.case_id]["expect"]["route"]
        observed = result.observed["route"]
        if expected == observed:
            correct += 1
            counts[expected]["tp"] += 1
        else:
            counts[observed]["fp"] = counts[observed]["fp"] + 1
            counts[expected]["fn"] = counts[expected]["fn"] + 1

    per_class = {}
    for name, c in counts.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **c,
        }

    return {
        "n": len(labelled),
        "accuracy": correct / len(labelled),
        "per_class": per_class,
    }


def crisis_metrics(results: list[CaseResult], cases_by_id: dict[str, dict]) -> dict:
    tp = fn = fp = tn = 0
    missed: list[str] = []
    false_alarms: list[str] = []

    for result in results:
        case = cases_by_id[result.case_id]
        expect = case.get("expect") or {}
        if "crisis" not in expect:
            continue
        observed = result.observed.get("crisis")
        if observed is None:
            turns = result.observed.get("turns") or []
            if not turns:
                continue
            observed = any(bool(t.get("crisis")) for t in turns)

        if expect["crisis"] and observed:
            tp += 1
        elif expect["crisis"] and not observed:
            fn += 1
            missed.append(result.case_id)
        elif not expect["crisis"] and observed:
            fp += 1
            false_alarms.append(result.case_id)
        else:
            tn += 1

    total_positive = tp + fn
    total_negative = fp + tn
    return {
        "recall": tp / total_positive if total_positive else None,
        "false_positive_rate": fp / total_negative if total_negative else None,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "missed_cases": missed,
        "false_alarm_cases": false_alarms,
    }


def verifier_metrics(results: list[CaseResult], cases_by_id: dict[str, dict]) -> dict:
    tp = fp = fn = tn = 0
    for result in results:
        case = cases_by_id[result.case_id]
        if case["category"] != "verifier_seeded":
            continue
        verdict = result.observed.get("verdict") or {}
        if "passed" not in verdict:
            continue
        should_reject = not (case.get("expect") or {}).get("verdict_passed", True)
        did_reject = not bool(verdict.get("passed"))
        if should_reject and did_reject:
            tp += 1
        elif not should_reject and did_reject:
            fp += 1
        elif should_reject and not did_reject:
            fn += 1
        else:
            tn += 1

    return {
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def latency_metrics(results: list[CaseResult]) -> dict:
    samples = sorted(r.latency_s for r in results if r.latency_s > 0 and not r.skipped)
    if not samples:
        return {}
    return {
        "n": len(samples),
        "p50_s": round(statistics.median(samples), 3),
        "p95_s": round(samples[min(int(len(samples) * 0.95), len(samples) - 1)], 3),
        "mean_s": round(statistics.fmean(samples), 3),
        "max_s": round(samples[-1], 3),
    }


def prometheus_snapshot() -> dict:
    from prometheus_client import REGISTRY

    wanted = {
        "llm_tokens_total",
        "llm_calls_total",
        "fallback_total",
        "verifier_verdicts_total",
        "tool_calls_total",
        "memory_ops_total",
        "crisis_detected_total",
        "agent_requests_total",
    }
    snapshot: dict[str, float] = {}
    for metric in REGISTRY.collect():
        if metric.name not in wanted:
            continue
        for sample in metric.samples:
            if sample.name.endswith("_created"):
                continue
            labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
            key = f"{sample.name}{{{labels}}}" if labels else sample.name
            snapshot[key] = sample.value
    return dict(sorted(snapshot.items()))


def summarise(results: list[CaseResult], cases_by_id: dict[str, dict]) -> dict:
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(
            result.category, {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        )
        if result.skipped:
            bucket["skipped"] += 1
        elif result.error:
            bucket["errors"] += 1
        elif result.passed:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1

    scored = [r for r in results if not r.skipped and not r.error]
    return {
        "cases_total": len(results),
        "cases_scored": len(scored),
        "cases_skipped": sum(1 for r in results if r.skipped),
        "cases_errored": sum(1 for r in results if r.error),
        "pass_rate": (
            sum(1 for r in scored if r.passed) / len(scored) if scored else None
        ),
        "by_category": by_category,
        "routing": routing_metrics(results, cases_by_id),
        "crisis": crisis_metrics(results, cases_by_id),
        "verifier": verifier_metrics(results, cases_by_id),
        "latency": latency_metrics(results),
        "counters": prometheus_snapshot(),
    }


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1%}" if suffix == "%" else f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def write_markdown(
    path: Path,
    summary: dict,
    results: list[CaseResult],
    cases_by_id: dict[str, dict],
    meta: dict,
) -> None:
    lines: list[str] = []
    lines.append("# Eval run\n")
    lines.append(f"- **Run:** {meta['run_id']}")
    lines.append(f"- **Mode:** {meta['mode']}")
    lines.append(f"- **Target:** {meta['target']}")
    lines.append(f"- **Vector search available:** {meta['vector_search']}")
    lines.append(f"- **Models:** " + ", ".join(
        f"{role}=`{model}`" for role, model in meta["models"].items()
    ))
    lines.append(f"- **Duration:** {meta['duration_s']:.1f}s\n")

    if meta.get("note"):
        lines.append(f"> {meta['note']}\n")

    if meta["mode"] == "offline":
        lines.append(
            "> Offline mode: crisis cases are scored on **detection only** "
            "(route and crisis flag). Whether the reply actually contains help "
            "resources needs a generating model and was not checked here.\n"
        )

    lines.append("## Headline\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Cases scored | {summary['cases_scored']} of {summary['cases_total']} |")
    lines.append(f"| End-to-end pass rate | {_fmt(summary['pass_rate'], '%')} |")
    routing = summary.get("routing") or {}
    lines.append(f"| Routing accuracy | {_fmt(routing.get('accuracy'), '%')} (n={routing.get('n', 0)}) |")
    crisis = summary.get("crisis") or {}
    lines.append(f"| **Crisis recall (target 100%)** | **{_fmt(crisis.get('recall'), '%')}** |")
    lines.append(f"| Crisis false-positive rate | {_fmt(crisis.get('false_positive_rate'), '%')} |")
    verifier = summary.get("verifier") or {}
    lines.append(f"| Verifier precision | {_fmt(verifier.get('precision'), '%')} |")
    lines.append(f"| Verifier recall | {_fmt(verifier.get('recall'), '%')} |")
    latency = summary.get("latency") or {}
    lines.append(f"| Latency p50 / p95 | {_fmt(latency.get('p50_s'), 's')} / {_fmt(latency.get('p95_s'), 's')} |")
    lines.append("")

    if crisis.get("missed_cases"):
        lines.append(
            f"**Crisis cases missed: {', '.join(crisis['missed_cases'])}** — "
            "this must be zero.\n"
        )
    if crisis.get("false_alarm_cases"):
        lines.append(
            f"Crisis false alarms: {', '.join(crisis['false_alarm_cases'])}\n"
        )

    lines.append("## By category\n")
    lines.append("| Category | Passed | Failed | Errors | Skipped |")
    lines.append("|---|---|---|---|---|")
    for category, counts in sorted(summary["by_category"].items()):
        lines.append(
            f"| {category} | {counts['passed']} | {counts['failed']} "
            f"| {counts['errors']} | {counts['skipped']} |"
        )
    lines.append("")

    if routing.get("per_class"):
        lines.append("## Routing, per class\n")
        lines.append("| Route | Precision | Recall | F1 | TP | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, c in routing["per_class"].items():
            lines.append(
                f"| {name} | {_fmt(c['precision'], '%')} | {_fmt(c['recall'], '%')} "
                f"| {_fmt(c['f1'], '%')} | {c['tp']} | {c['fp']} | {c['fn']} |"
            )
        lines.append("")

    failed = [r for r in results if not r.passed and not r.skipped]
    if failed:
        lines.append("## Failures\n")
        for result in failed:
            note = cases_by_id[result.case_id].get("note")
            lines.append(f"### `{result.case_id}` ({result.category})\n")
            if result.error:
                lines.append(f"- Error: `{result.error}`")
            for failure in result.failures:
                lines.append(f"- {failure}")
            if note:
                lines.append(f"- Dataset note: {note}")
            lines.append("")

    skipped = [r for r in results if r.skipped]
    if skipped:
        lines.append("## Skipped\n")
        for result in skipped:
            lines.append(f"- `{result.case_id}`: {result.skipped}")
        lines.append("")

    rubric = [
        (r, cases_by_id[r.case_id])
        for r in results
        if cases_by_id[r.case_id].get("rubric")
    ]
    if rubric:
        lines.append("## Manual rubric sheet\n")
        lines.append(
            "No LLM-as-judge is used, so these replies need a human verdict. "
            "Tick each criterion.\n"
        )
        for result, case in rubric:
            turns = result.observed.get("turns") or []
            reply = turns[-1].get("reply", "") if turns else ""
            lines.append(f"### `{case['id']}`\n")
            lines.append(f"**Message:** {case.get('message', '')}\n")
            lines.append(f"**Reply:**\n\n> {reply.replace(chr(10), chr(10) + '> ')}\n")
            for criterion in case["rubric"]:
                lines.append(f"- [ ] {criterion}")
            lines.append("")

    counters = summary.get("counters") or {}
    if counters:
        lines.append("## Counters from this run\n")
        lines.append("```")
        for key, value in counters.items():
            lines.append(f"{key} {value:g}")
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the task-blocks eval suite.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only cases that need no remote model (routing, verifier rules).",
    )
    parser.add_argument("--category", action="append", help="Category to include.")
    parser.add_argument("--limit", type=int, help="Maximum number of cases.")
    parser.add_argument("--api", help="Run against a running API instead of in-process.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=float(os.getenv("EVAL_SLEEP_SECONDS", "1.0")),
        help="Pause between remote cases, to stay inside free-tier limits.",
    )
    parser.add_argument("--user", default="eval-user", help="Base user_id.")
    parser.add_argument("--note", default="", help="Note recorded in the report.")
    args = parser.parse_args()

    from task_blocks.config import get_settings
    from task_blocks.logging_config import configure_logging
    from task_blocks.memory import sqlite_supports_extensions
    from task_blocks.observability import set_build_info

    configure_logging(os.getenv("LOG_LEVEL", "WARNING"))
    set_build_info()
    settings = get_settings()

    categories = set(args.category) if args.category else None
    if args.offline:
        categories = (categories or OFFLINE_CATEGORIES) & OFFLINE_CATEGORIES

    cases = load_cases(DATASET, categories=categories, limit=args.limit)
    if not cases:
        print("No cases selected.", file=sys.stderr)
        return 2
    cases_by_id = {case["id"]: case for case in cases}

    runner: Any = ApiRunner(args.api) if args.api else GraphRunner()
    remote_categories = {"crisis", "tool_use", "memory", "prompt_injection", "quality_rubric"}

    run_id = next_run_id("run")
    started = time.perf_counter()
    results: list[CaseResult] = []

    print(f"Running {len(cases)} cases ({'api' if args.api else 'in-process'})\n")
    for index, case in enumerate(cases, start=1):
        case_id = case["id"]
        category = case["category"]
        print(f"[{index}/{len(cases)}] {case_id} ({category}) ... ", end="", flush=True)

        try:
            if category == "verifier_seeded":
                result = run_seeded_verifier_case(runner, case)
            elif category in ROUTE_ONLY_CATEGORIES or (
                args.offline and category == "crisis"
            ):
                result = run_routing_case(runner, case)
                if args.offline and category == "crisis":
                    result.checks["reply_checks"] = "not applied (offline mode)"
            else:
                result = run_turn_case(runner, case, user_id=args.user)
                if args.sleep and category in remote_categories:
                    time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            result = CaseResult(
                case_id=case_id,
                category=category,
                passed=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        results.append(result)
        if result.skipped:
            print(f"skipped ({result.skipped})")
        elif result.error:
            print(f"ERROR {result.error[:80]}")
        elif result.passed:
            print(f"pass ({result.latency_s:.1f}s)")
        else:
            print(f"FAIL {result.failures[0][:80] if result.failures else ''}")

    duration = time.perf_counter() - started
    summary = summarise(results, cases_by_id)
    meta = {
        "run_id": run_id,
        "duration_s": duration,
        "mode": "offline" if args.offline else "full",
        "target": args.api or "in-process graph",
        "vector_search": sqlite_supports_extensions(),
        "models": {
            role: cfg.model_id for role, cfg in settings.models_by_role().items()
        },
        "note": args.note,
        "selected_categories": sorted(categories) if categories else "all",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{run_id}.json"
    md_path = RESULTS_DIR / f"{run_id}.md"

    json_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "summary": summary,
                "cases": [
                    {
                        "id": r.case_id,
                        "category": r.category,
                        "passed": r.passed,
                        "failures": r.failures,
                        "error": r.error,
                        "skipped": r.skipped,
                        "latency_s": round(r.latency_s, 3),
                        "observed": r.observed,
                    }
                    for r in results
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown(md_path, summary, results, cases_by_id, meta)

    print("\n" + "=" * 60)
    print(f"pass rate       {_fmt(summary['pass_rate'], '%')} "
          f"({summary['cases_scored']} scored)")
    if summary.get("routing"):
        print(f"routing acc     {_fmt(summary['routing'].get('accuracy'), '%')}")
    if summary.get("crisis", {}).get("recall") is not None:
        print(f"crisis recall   {_fmt(summary['crisis']['recall'], '%')} "
              f"(target 100%)")
    if summary.get("verifier", {}).get("recall") is not None:
        print(f"verifier P/R    {_fmt(summary['verifier'].get('precision'), '%')}"
              f" / {_fmt(summary['verifier'].get('recall'), '%')}")
    print(f"\nwrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")


    if summary.get("crisis", {}).get("missed_cases"):
        print("\nFAILED: crisis cases were missed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
