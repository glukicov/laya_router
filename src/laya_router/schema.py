"""The output contract shared by every backend.

Both backends answer the same questions, so both return the same object. That is the point of the
repo: swapping a 421M local encoder for a hosted LLM should be a configuration change, not a
rewrite of everything downstream of the classifier.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from laya_router.questions import QUESTION_IDS


class Decision(BaseModel):
    """One question's answer, with the backend's own confidence in it.

    `confidence` means different things on each side, and that difference is a result rather than a
    wart: Laya reports a temperature-scaled probability from its decision head, while a generative
    classifier reports a number it wrote itself. The evaluation measures both against outcomes.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)


class RouteResult(BaseModel):
    """Every decision for one request, plus what the routing call cost in time and money."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["laya", "openai"]
    model: str
    decisions: dict[str, Decision]
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)

    def answers(self) -> dict[str, str]:
        """Just the answers, in schema order, for scoring against the gold labels."""
        return {qid: self.decisions[qid].answer for qid in QUESTION_IDS if qid in self.decisions}

    def to_row(self) -> dict[str, Any]:
        """Flatten to one JSONL row, keeping per-question confidence for the calibration analysis."""
        row: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }
        for qid, decision in self.decisions.items():
            row[f"{qid}_answer"] = decision.answer
            row[f"{qid}_confidence"] = decision.confidence
        return row
