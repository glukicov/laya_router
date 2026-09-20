"""Both backends must produce the same shape from very different raw outputs."""

import json

import pytest

from laya_router.backends.laya_backend import LayaBackend, _decision
from laya_router.backends.openai_backend import OpenAIBackend, _clamp_confidence, price_for
from laya_router.questions import TIERS
from tests.fakes import FakeOpenAIClient


def test_laya_choice_confidence_is_the_probability_of_the_chosen_option() -> None:
    """Not the SDK's `confidence` field, which for a choice question is normalised entropy.

    Entropy answers "how peaked is this distribution"; the evaluation needs "how likely is this answer to be
    right", because that is what the generative classifier reports and what the calibration analysis scores.
    """
    decision = _decision(
        "tier",
        {
            "type": "choice",
            "choice": "medium",
            "confidence": 0.13,
            "probabilities": {"small": 0.25, "medium": 0.6, "powerful": 0.15},
        },
    )

    assert decision.answer == "medium"
    assert decision.confidence == pytest.approx(0.6)


def test_laya_noul_answer_becomes_a_true_false_decision() -> None:
    decision = _decision("needs_tools", {"type": "noul", "noul": 0.82, "confidence": 0.82})

    assert decision.answer == "true"
    assert decision.confidence == pytest.approx(0.82)
    assert decision.probabilities == {"true": pytest.approx(0.82), "false": pytest.approx(0.18)}


def test_laya_noul_below_the_coin_flip_reports_confidence_in_false() -> None:
    decision = _decision("needs_tools", {"type": "noul", "noul": 0.1, "confidence": 0.9})

    assert decision.answer == "false"
    # Confidence is the distance from 0.5, not P(true), so a firm 'no' is a confident answer.
    assert decision.confidence == pytest.approx(0.9)


def test_laya_backend_uses_a_single_forward_pass_per_message() -> None:
    class StubAgent:
        device = "cpu"

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, state: dict[str, str], questions: dict[str, object]) -> dict[str, object]:
            del state, questions
            self.calls += 1
            return {
                "answers": {
                    "tier": {
                        "type": "choice",
                        "choice": "medium",
                        "confidence": 0.7,
                        "probabilities": dict.fromkeys(TIERS, 0.1) | {"medium": 0.7},
                    },
                    "needs_tools": {"type": "noul", "noul": 0.2, "confidence": 0.8},
                    "is_sensitive": {"type": "noul", "noul": 0.9, "confidence": 0.9},
                },
                "usage": {"input_tokens": 120, "output_tokens": 0},
            }

    agent = StubAgent()
    result = LayaBackend(agent=agent).classify("I was charged twice.")

    assert agent.calls == 1
    assert result.backend == "laya"
    assert result.answers() == {"tier": "medium", "needs_tools": "false", "is_sensitive": "true"}
    # Nothing is generated, so there is nothing to bill for output tokens.
    assert result.output_tokens == 0
    assert result.cost_usd == 0.0


def test_openai_backend_prices_the_call_from_reported_usage() -> None:
    payload = json.dumps(
        {
            "tier": "powerful",
            "tier_confidence": 0.88,
            "needs_tools": True,
            "needs_tools_confidence": 0.91,
            "is_sensitive": False,
            "is_sensitive_confidence": 0.6,
        }
    )
    client = FakeOpenAIClient(payload, prompt_tokens=1_000_000, completion_tokens=1_000_000)

    result = OpenAIBackend(model="gpt-5-nano", client=client).classify("500s everywhere")

    assert result.answers() == {"tier": "powerful", "needs_tools": "true", "is_sensitive": "false"}
    # One million of each token at $0.05 in / $0.40 out.
    assert result.cost_usd == pytest.approx(0.45)
    assert client.completions.calls[0]["response_format"]["json_schema"]["strict"] is True


def test_openai_schema_offers_exactly_the_tiers_laya_scores() -> None:
    client = FakeOpenAIClient(
        json.dumps(
            {
                "tier": "small",
                "tier_confidence": 0.5,
                "needs_tools": False,
                "needs_tools_confidence": 0.5,
                "is_sensitive": False,
                "is_sensitive_confidence": 0.5,
            }
        )
    )

    OpenAIBackend(model="gpt-5-nano", client=client).classify("hello")

    schema = client.completions.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["tier"]["enum"] == list(TIERS)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.9, 0.9), (95, 0.95), (1.0, 1.0), (-2, 0.0), (250, 1.0), ("nonsense", 0.5), (None, 0.5)],
)
def test_self_reported_confidence_is_clamped_into_a_probability(raw: object, expected: float) -> None:
    assert _clamp_confidence(raw) == pytest.approx(expected)


def test_price_lookup_accepts_dated_model_snapshots() -> None:
    assert price_for("gpt-5-nano-2025-08-07") == price_for("gpt-5-nano")
    with pytest.raises(KeyError):
        price_for("some-other-vendor-model")
