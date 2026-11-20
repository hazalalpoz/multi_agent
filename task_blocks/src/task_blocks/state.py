from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Route = Literal["emotional", "logical"]


class RouterDecision(BaseModel):
    route: Route = Field(description="Which specialist should handle the turn.")
    crisis: bool = Field(
        description="True if the message signals risk to life or safety."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the route, not in crisis."
    )


class VerifierDecision(BaseModel):
    passed: bool = Field(alias="pass", description="True if the draft may be sent.")
    reasons: list[str] = Field(
        default_factory=list, description="One short sentence per problem found."
    )

    model_config = {"populate_by_name": True}


class Verdict(TypedDict):
    passed: bool
    reasons: list[str]
    source: Literal["rules", "llm", "rules+llm", "none"]


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    route: Route
    crisis: bool
    confidence: float
    skill_hint: str | None

    draft: str
    verdict: Verdict
    attempts: int
    model_used: str
    provider_error: bool

    user_id: str
    thread_id: str
    memories: list[str]


def initial_state(
    message: str, *, user_id: str, thread_id: str
) -> dict[str, object]:
    return {
        "messages": [{"role": "user", "content": message}],
        "user_id": user_id,
        "thread_id": thread_id,
        "attempts": 0,
        "crisis": False,
        "skill_hint": None,
        "memories": [],
    }
