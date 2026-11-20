from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from task_blocks.graph import (
    SAFE_FALLBACK_CRISIS_REPLY,
    SAFE_FALLBACK_PROVIDER_REPLY,
    SAFE_FALLBACK_REPLY,
    _history_for_model,
    _last_user_text,
    _memories_block,
    _parse_json_object,
    build_graph,
    route_from_router,
    route_from_verifier,
    router,
    safe_fallback,
    verifier,
)


def test_graph_has_every_declared_node() -> None:
    nodes = set(build_graph(persist=False).get_graph().nodes)
    assert {
        "load_memory",
        "router",
        "therapist",
        "logical",
        "verifier",
        "regenerate",
        "safe_fallback",
        "save_turn",
    } <= nodes


@pytest.mark.parametrize(
    ("route", "expected"),
    [("emotional", "therapist"), ("logical", "logical"), (None, "logical")],
)
def test_route_from_router(route: str | None, expected: str) -> None:
    assert route_from_router({"route": route}) == expected


def test_verifier_pass_goes_to_save_turn() -> None:
    state = {"verdict": {"passed": True, "reasons": [], "source": "llm"}}
    assert route_from_verifier(state) == "save_turn"


def test_verifier_fail_first_attempt_regenerates() -> None:
    state = {
        "verdict": {"passed": False, "reasons": ["bad"], "source": "rules"},
        "attempts": 0,
    }
    assert route_from_verifier(state) == "regenerate"


def test_verifier_fail_after_retry_goes_to_safe_fallback() -> None:
    state = {
        "verdict": {"passed": False, "reasons": ["still bad"], "source": "rules"},
        "attempts": 1,
    }
    assert route_from_verifier(state) == "safe_fallback"


def test_regeneration_loop_cannot_run_forever() -> None:
    for attempts in range(1, 6):
        state = {
            "verdict": {"passed": False, "reasons": ["x"], "source": "rules"},
            "attempts": attempts,
        }
        assert route_from_verifier(state) == "safe_fallback"


def test_router_falls_back_to_logical_when_the_model_fails() -> None:
    state = {"messages": [HumanMessage(content="What is 2 + 2?")]}
    with patch("task_blocks.graph.get_model", side_effect=RuntimeError("ollama down")):
        result = router(state)
    assert result["route"] == "logical"
    assert result["crisis"] is False
    assert result["confidence"] == 0.0


def test_router_fallback_still_catches_crisis_by_keyword() -> None:
    state = {"messages": [HumanMessage(content="I want to kill myself")]}
    with patch("task_blocks.graph.get_model", side_effect=RuntimeError("ollama down")):
        result = router(state)
    assert result["crisis"] is True
    assert result["route"] == "emotional"
    assert result["skill_hint"] == "crisis-response"


def test_keyword_crisis_overrides_a_model_that_says_otherwise() -> None:
    from task_blocks.state import RouterDecision

    class StubModel:
        def with_structured_output(self, *_args, **_kwargs):
            return self

        def invoke(self, _messages):
            return RouterDecision(route="logical", crisis=False, confidence=0.99)

    state = {"messages": [HumanMessage(content="honestly I want to die")]}
    with patch("task_blocks.graph.get_model", return_value=StubModel()):
        result = router(state)
    assert result["crisis"] is True
    assert result["route"] == "emotional"


def test_verifier_rules_override_a_passing_model() -> None:
    from task_blocks.state import VerifierDecision

    class StubModel:
        def with_structured_output(self, *_args, **_kwargs):
            return self

        def invoke(self, _messages):
            return VerifierDecision.model_validate({"pass": True, "reasons": []})

    state = {
        "messages": [HumanMessage(content="I want to kill myself")],
        "draft": "That sounds really hard. Tell me more.",
        "route": "emotional",
        "crisis": True,
        "attempts": 0,
    }
    with patch("task_blocks.graph.get_model", return_value=StubModel()):
        verdict = verifier(state)["verdict"]
    assert verdict["passed"] is False
    assert verdict["source"] == "rules+llm"
    assert any("help resources" in r for r in verdict["reasons"])


def test_verifier_passes_when_model_is_unavailable_and_rules_are_clean() -> None:
    state = {
        "messages": [HumanMessage(content="What is the capital of Portugal?")],
        "draft": "Lisbon.",
        "route": "logical",
        "crisis": False,
        "attempts": 0,
    }
    with patch("task_blocks.graph.get_model", side_effect=RuntimeError("down")):
        verdict = verifier(state)["verdict"]
    assert verdict["passed"] is True


def test_verifier_fails_empty_draft_without_calling_a_model() -> None:
    state = {
        "messages": [HumanMessage(content="hello")],
        "draft": "",
        "route": "logical",
        "crisis": False,
    }
    with patch("task_blocks.graph.get_model", side_effect=AssertionError("called")):
        verdict = verifier(state)["verdict"]
    assert verdict["passed"] is False
    assert verdict["source"] == "rules"


def test_safe_fallback_crisis_reply_contains_resources() -> None:
    result = safe_fallback({"crisis": True})
    reply = result["messages"][0].content
    assert reply == SAFE_FALLBACK_CRISIS_REPLY
    assert "112" in reply
    assert "findahelpline.com" in reply


def test_safe_fallback_crisis_reply_passes_its_own_rules() -> None:
    from task_blocks.verification import check_reply

    assert check_reply(SAFE_FALLBACK_CRISIS_REPLY, route="emotional", crisis=True) == []
    assert check_reply(SAFE_FALLBACK_REPLY, route="logical", crisis=False) == []


def test_safe_fallback_blames_the_provider_not_the_user() -> None:
    result = safe_fallback({"crisis": False, "provider_error": True})
    reply = result["messages"][0].content
    assert reply == SAFE_FALLBACK_PROVIDER_REPLY
    assert "not with your question" in reply
    assert "rephrasing" not in reply


def test_safe_fallback_provider_reply_passes_its_own_rules() -> None:
    from task_blocks.verification import check_reply

    assert check_reply(
        SAFE_FALLBACK_PROVIDER_REPLY, route="logical", crisis=False
    ) == []


def test_crisis_outranks_a_provider_error() -> None:
    result = safe_fallback({"crisis": True, "provider_error": True})
    reply = result["messages"][0].content
    assert reply == SAFE_FALLBACK_CRISIS_REPLY
    assert "112" in reply


def test_safe_fallback_non_crisis_reply_is_honest() -> None:
    result = safe_fallback({"crisis": False})
    assert result["messages"][0].content == SAFE_FALLBACK_REPLY
    assert result["model_used"] == "safe-fallback"


def test_last_user_text_reads_message_objects() -> None:
    messages = [HumanMessage(content="first"), AIMessage(content="reply"),
                HumanMessage(content="second")]
    assert _last_user_text(messages) == "second"


def test_last_user_text_reads_raw_dicts() -> None:
    assert _last_user_text([{"role": "user", "content": "hi"}]) == "hi"


def test_last_user_text_handles_no_user_message() -> None:
    assert _last_user_text([AIMessage(content="only me")]) == ""
    assert _last_user_text([]) == ""


def test_history_is_trimmed_to_the_configured_window() -> None:
    from task_blocks.config import get_settings

    limit = get_settings().history_max_messages
    messages: list = []
    for i in range(limit * 3):
        messages.append(HumanMessage(content=f"q{i}"))
        messages.append(AIMessage(content=f"a{i}"))

    trimmed = _history_for_model(messages)
    assert len(trimmed) <= limit
    assert isinstance(trimmed[0], HumanMessage), "window must open on a human turn"
    assert trimmed[-1].content == messages[-1].content


def test_memories_block_is_empty_without_memories() -> None:
    assert _memories_block([]) == ""
    assert _memories_block(None) == ""


def test_memories_block_states_the_facts_are_known() -> None:
    block = _memories_block(["Works as a landscape architect"])
    assert "Works as a landscape architect" in block
    assert "know" in block.lower()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"route": "logical"}', {"route": "logical"}),
        ('```json\n{"route": "logical"}\n```', {"route": "logical"}),
        ('Sure! {"route": "logical"} hope that helps', {"route": "logical"}),
        ("not json at all", None),
        ("", None),
        ("[1, 2, 3]", None),
    ],
)
def test_parse_json_object(raw: str, expected: dict | None) -> None:
    assert _parse_json_object(raw) == expected
