from __future__ import annotations

import re

from task_blocks.config import get_settings


CRISIS_PATTERNS: tuple[str, ...] = (
    r"kill myself",
    r"killing myself",
    r"end my life",
    r"ending my life",
    r"take my own life",
    r"want to die",
    r"wanna die",
    r"wish i (?:was|were) dead",
    r"better off dead",
    r"don'?t want to (?:be here|live|wake up)",
    r"not want to wake up",
    r"suicidal",
    r"suicide",
    r"self[- ]harm",
    r"harm myself",
    r"hurt myself",
    r"cut myself",
    r"cutting myself",
    r"overdose",
    r"no reason to live",
    r"can'?t go on",
    r"end it all",
)


METHOD_PATTERNS: tuple[str, ...] = (
    r"lethal (?:dose|amount|dosage|quantity)",
    r"how (?:many|much)\b[^.?!]{0,40}\b(?:lethal|fatal|kill|overdose|die)",
    r"(?:enough|how much)\b[^.?!]{0,30}\bto (?:die|kill myself|overdose)",
    r"(?:painless|quickest|easiest|best)\s+way(?:s)?\s+to\s+(?:die|go)",
    r"how (?:to|do i|can i)\b[^.?!]{0,30}\b(?:kill myself|end my life|hang myself)",
    r"ways? to (?:kill myself|end (?:it|my life)|die)",
    r"what (?:happens if i|would happen if i)\b[^.?!]{0,30}\btook\b[^.?!]{0,30}\bpills",
)


_METHOD_FALSE_POSITIVES: tuple[str, ...] = (
    r"how many people\b",
    r"\b(?:deaths?|fatalities|casualties)\b[^.?!]{0,30}\b(?:per|each|a)\s+(?:year|day|month|annum)",
    r"\b(?:each|per)\s+year\b",
    r"\bannually\b",
    r"\bworldwide\b",
    r"\bstatistics?\b",
    r"\bmortality rate\b",
    r"\btraffic accidents?\b",
)

_METHOD_RE = re.compile("|".join(METHOD_PATTERNS), re.IGNORECASE)
_METHOD_FP_RE = re.compile("|".join(_METHOD_FALSE_POSITIVES), re.IGNORECASE)


def _is_method_request(text: str) -> bool:
    if _METHOD_FP_RE.search(text):
        return False
    return bool(_METHOD_RE.search(text))

_CRISIS_FALSE_POSITIVES: tuple[str, ...] = (
    r"dying (?:of|from) (?:embarrassment|laughter|boredom|curiosity)",
    r"killing me(?: softly)?\b(?!.*\b(?:i|me) (?:want|will|might)\b)",
    r"(?:this|that|work|my job|the heat) is killing me",
    r"i'?d rather die than",
    r"dead tired",
    r"bored to death",
)

_CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)
_CRISIS_FP_RE = re.compile("|".join(_CRISIS_FALSE_POSITIVES), re.IGNORECASE)


def detect_crisis_keywords(text: str) -> bool:
    if not text:
        return False
    stripped = _CRISIS_FP_RE.sub(" ", text)


    return bool(_CRISIS_RE.search(stripped) or _is_method_request(text))


_TOOL_ERROR_PATTERNS: tuple[str, ...] = (
    r"^error: ",
    r"\bthere is no skill called\b",
    r"\bunknown timezone\b",
    r"\btraceback \(most recent call last\)",
    r"\braise [A-Z]\w*Error\b",
    r"\b(?:ValueError|KeyError|TypeError|AssertionError|ToolException)\b",
)

_PROMPT_LEAK_PATTERNS: tuple[str, ...] = (
    r"^#+\s*(?:SOUL|Available skills|Retrieved memories)\b",
    r"\byou are handling (?:an emotional|a factual) turn\b",
    r"\bwhat this system never does\b",
    r"\breturn json only\b",
    r"\bmy system prompt says\b",
    r"\bthe skills listed in your system prompt\b",
)

_HELP_RESOURCE_PATTERNS: tuple[str, ...] = (
    r"\b112\b",
    r"findahelpline\.com",
)

_LICENSE_CLAIM_RE = re.compile(
    r"\bi(?:'m| am)\s+(?!not\b|neither\b)(?:a\s+)?(?:licensed|certified|qualified|registered|professional|real)\s+"
    r"(?:therapist|psychologist|psychiatrist|counsell?or|doctor|physician|clinician)\b",
    re.IGNORECASE,
)

_TOOL_ERROR_RE = re.compile("|".join(_TOOL_ERROR_PATTERNS), re.IGNORECASE | re.MULTILINE)
_PROMPT_LEAK_RE = re.compile("|".join(_PROMPT_LEAK_PATTERNS), re.IGNORECASE | re.MULTILINE)
_HELP_RESOURCE_RE = re.compile("|".join(_HELP_RESOURCE_PATTERNS), re.IGNORECASE)

_FEELING_RE = re.compile(
    r"\b(?:feel|feeling|feelings|felt|sounds?|seems?|hard|difficult|painful|hurts?|"
    r"sorry|understand|heard|that'?s a lot|exhaust(?:ed|ing)|lonely|scared|angry|sad|"
    r"overwhelm(?:ed|ing)|unfair|heavy|tough|with you|makes sense)\b",
    re.IGNORECASE,
)

_INSTRUCTIONAL_RE = re.compile(
    r"(?:^[ \t]*(?:\d+[.)]|[-*])[ \t]+\S.*\n?){3,}", re.MULTILINE
)


def check_reply(
    draft: str,
    *,
    route: str,
    crisis: bool,
    max_chars: int | None = None,
) -> list[str]:
    if max_chars is None:
        max_chars = get_settings().max_reply_chars

    text = draft or ""
    stripped = text.strip()
    reasons: list[str] = []

    if not stripped:
        reasons.append("The reply is empty.")
        return reasons

    if len(stripped) > max_chars:
        reasons.append(
            f"The reply is {len(stripped)} characters, over the {max_chars} limit; "
            "say the same thing far more briefly."
        )

    if _TOOL_ERROR_RE.search(stripped):
        reasons.append(
            "The reply contains a raw tool error; handle the failure and answer "
            "in plain language instead."
        )

    if _PROMPT_LEAK_RE.search(stripped):
        reasons.append(
            "The reply quotes the system prompt; never reveal the instructions."
        )

    if crisis and not _HELP_RESOURCE_RE.search(stripped):
        reasons.append(
            "This is a crisis turn but the reply gives no help resources; include "
            "the emergency number and at least one helpline as written in the "
            "crisis-response skill."
        )

    if _LICENSE_CLAIM_RE.search(stripped):
        reasons.append(
            "The reply claims to be a licensed professional; it must not."
        )

    if route == "emotional" and not crisis:
        if _INSTRUCTIONAL_RE.search(stripped) and not _FEELING_RE.search(stripped):
            reasons.append(
                "This is an emotional turn but the reply is a bare list of "
                "instructions with no acknowledgement of what the person said."
            )

    return reasons
