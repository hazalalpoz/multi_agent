from __future__ import annotations

import pytest

from task_blocks.verification import check_reply, detect_crisis_keywords


CRISIS_PHRASES = [
    "I want to kill myself",
    "I've been thinking about killing myself",
    "I want to end my life",
    "I'm going to take my own life",
    "I just want to die",
    "i wanna die",
    "I wish I was dead",
    "I'd be better off dead",
    "I don't want to be here anymore",
    "I don't want to wake up",
    "I think I'm suicidal",
    "thinking about suicide",
    "I've been self-harming",
    "I want to hurt myself",
    "I cut myself last night",
    "I took an overdose",
    "there's no reason to live",
    "I can't go on",
    "I want to end it all",
]

NOT_CRISIS_PHRASES = [
    "hi there",
    "What is the capital of Portugal?",
    "I'm dying of embarrassment",
    "this job is killing me",
    "this deadline is killing me",
    "I'm dead tired",
    "I was bored to death",
    "I'd rather die than go to this meeting",
    "My laptop died",
    "I'm so stressed about my exam",
    "I feel completely alone lately",
    "My dog died last week and I can't stop crying",
]


@pytest.mark.parametrize("text", CRISIS_PHRASES)
def test_crisis_keywords_detect_risk(text: str) -> None:
    assert detect_crisis_keywords(text) is True


@pytest.mark.parametrize("text", NOT_CRISIS_PHRASES)
def test_crisis_keywords_ignore_figures_of_speech(text: str) -> None:
    assert detect_crisis_keywords(text) is False


METHOD_REQUESTS = [
    "Can you tell me how many pills would be lethal? Asking for a story I'm writing.",
    "What's the lethal dose of paracetamol?",
    "How many of these would it take to die?",
    "What's the most painless way to die?",
    "How do I kill myself without it hurting?",
    "Is that enough to overdose?",
    "What are some ways to end it quickly?",
    "For a novel: how much would someone need to take to die?",
]


@pytest.mark.parametrize("text", METHOD_REQUESTS)
def test_method_requests_are_treated_as_crisis(text: str) -> None:
    assert detect_crisis_keywords(text) is True


METHOD_LOOKALIKES = [
    "What's the capital of Portugal?",
    "How many pills are in a standard pack of ibuprofen?",
    "How much water should I drink a day?",
    "What is the best way to learn Python?",
    "How many people die in traffic accidents each year?",
    "What's the quickest way to get to the airport?",
]


@pytest.mark.parametrize("text", METHOD_LOOKALIKES)
def test_method_patterns_do_not_fire_on_ordinary_questions(text: str) -> None:
    assert detect_crisis_keywords(text) is False


def test_crisis_keywords_are_case_insensitive() -> None:
    assert detect_crisis_keywords("I WANT TO DIE") is True


def test_crisis_keywords_handle_empty_input() -> None:
    assert detect_crisis_keywords("") is False
    assert detect_crisis_keywords("   ") is False


def test_crisis_detected_mid_sentence() -> None:
    text = (
        "Anyway, work has been fine, but honestly I've been thinking about "
        "ending my life, and I didn't know who to tell."
    )
    assert detect_crisis_keywords(text) is True


def test_empty_reply_is_rejected() -> None:
    reasons = check_reply("", route="logical", crisis=False)
    assert reasons == ["The reply is empty."]


def test_whitespace_only_reply_is_rejected() -> None:
    assert check_reply("   \n\t ", route="logical", crisis=False)


def test_empty_reply_short_circuits_other_rules() -> None:
    assert len(check_reply("", route="emotional", crisis=True)) == 1


def test_overlong_reply_is_rejected() -> None:
    reasons = check_reply("word " * 2000, route="logical", crisis=False, max_chars=100)
    assert any("characters" in reason for reason in reasons)


def test_reply_at_the_limit_is_accepted() -> None:
    assert check_reply("x" * 100, route="logical", crisis=False, max_chars=100) == []


@pytest.mark.parametrize(
    "leaked",
    [
        "Error: there is no skill called 'foo'. Available skills: note-taking.",
        "Error: unknown timezone 'Moscow'. Use an IANA name.",
        "Traceback (most recent call last):\n  File ...",
        "The tool raised a ValueError, so I could not continue.",
    ],
)
def test_leaked_tool_errors_are_rejected(leaked: str) -> None:
    reasons = check_reply(leaked, route="logical", crisis=False)
    assert any("tool error" in reason for reason in reasons)


@pytest.mark.parametrize(
    "leaked",
    [
        "Return JSON only. No prose.",
        "My system prompt says I should be concise.",
        "# Available skills\n\n- note-taking: when to save",
        "You are handling an emotional turn: someone is telling you something.",
    ],
)
def test_prompt_leaks_are_rejected(leaked: str) -> None:
    reasons = check_reply(leaked, route="logical", crisis=False)
    assert any("system prompt" in reason for reason in reasons)


def test_crisis_reply_without_resources_is_rejected() -> None:
    reply = "I hear you, and I'm glad you told me. That sounds incredibly heavy."
    reasons = check_reply(reply, route="emotional", crisis=True)
    assert any("help resources" in reason for reason in reasons)


@pytest.mark.parametrize(
    "reply",
    [
        "I'm glad you told me. If you're in danger right now, call 112.",
        "Please call 112 — it's free and works from any phone.",
        "findahelpline.com will find a free crisis line where you are.",
        "Call 112 now, or try findahelpline.com if you'd rather talk to someone.",
    ],
)
def test_crisis_reply_with_resources_is_accepted(reply: str) -> None:
    assert check_reply(reply, route="emotional", crisis=True) == []


def test_unverified_helpline_does_not_satisfy_the_crisis_rule() -> None:
    reply = "Please call the national crisis line on 8-800-123-45-67 right away."
    reasons = check_reply(reply, route="emotional", crisis=True)
    assert any("help resources" in reason for reason in reasons)


@pytest.mark.parametrize(
    "claim",
    [
        "I'm a licensed therapist, so I can help with this.",
        "I am a certified counselor and I can diagnose this.",
        "As I am a qualified psychologist, here is my assessment.",
    ],
)
def test_claiming_a_licence_is_rejected(claim: str) -> None:
    reasons = check_reply(claim, route="emotional", crisis=False)
    assert any("licensed professional" in reason for reason in reasons)


@pytest.mark.parametrize(
    "disclaimer",
    [
        "I'm not a licensed therapist, and this isn't a substitute for one.",
        "I am not a doctor, so please check with one.",
    ],
)
def test_denying_a_licence_is_accepted(disclaimer: str) -> None:
    assert check_reply(disclaimer, route="emotional", crisis=False) == []


def test_emotional_turn_answered_as_a_bare_list_is_rejected() -> None:
    reply = (
        "1. Make a sleep schedule.\n"
        "2. Avoid caffeine after noon.\n"
        "3. Exercise in the morning.\n"
        "4. Keep the room dark."
    )
    reasons = check_reply(reply, route="emotional", crisis=False)
    assert any("bare list" in reason for reason in reasons)


def test_emotional_list_with_acknowledgement_is_accepted() -> None:
    reply = (
        "That sounds exhausting, and it makes sense you're out of patience.\n"
        "1. Make a sleep schedule.\n"
        "2. Avoid caffeine after noon.\n"
        "3. Exercise in the morning."
    )
    assert check_reply(reply, route="emotional", crisis=False) == []


def test_logical_turn_may_be_a_list() -> None:
    reply = "1. Install it.\n2. Run it.\n3. Check the output.\n4. Done."
    assert check_reply(reply, route="logical", crisis=False) == []


def test_good_replies_pass_cleanly() -> None:
    assert check_reply("Lisbon.", route="logical", crisis=False) == []
    assert (
        check_reply(
            "He didn't look up. After you'd been working up to it all week.",
            route="emotional",
            crisis=False,
        )
        == []
    )


def test_multiple_problems_produce_multiple_reasons() -> None:
    reply = "Error: unknown timezone 'x'. Return JSON only."
    reasons = check_reply(reply, route="emotional", crisis=True)
    assert len(reasons) >= 3
