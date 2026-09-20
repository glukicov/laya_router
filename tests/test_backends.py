"""Both backends must produce the same shape from very different raw outputs."""

import json

import pytest

from laya_router.backends.laya_backend import LayaBackend, _decision
from laya_router.backends.openai_backend import OpenAIBackend, _clamp_confidence, price_for
from laya_router.questions import QUEUES
from tests.fakes import FakeOpenAIClient


def test_laya_noul_answer_becomes_a_true_false_decision() -> None:
    decision = _decision("urgent", {"type": "noul", "noul": 0.82, "confidence": 0.82})

    assert decision.answer == "true"
    assert decision.confidence == pytest.approx(0.82)
    assert decision.probabilities == {"true": pytest.approx(0.82), "false": pytest.approx(0.18)}


def test_laya_noul_below_the_coin_flip_reports_confidence_in_false() -> None:
    decision = _decision("urgent", {"type": "noul", "noul": 0.1, "confidence": 0.9})

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
                    "queue": {
                        "type": "choice",
                        "choice": "billing",
                        "confidence": 0.7,
                        "probabilities": dict.fromkeys(QUEUES, 0.1) | {"billing": 0.7},
                    },
                    "urgent": {"type": "noul", "noul": 0.2, "confidence": 0.8},
                    "needs_human": {"type": "noul", "noul": 0.9, "confidence": 0.9},
                },
                "usage": {"input_tokens": 120, "output_tokens": 0},
            }

    agent = StubAgent()
    result = LayaBackend(agent=agent).classify("I was charged twice.")

    assert agent.calls == 1
    assert result.backend == "laya"
    assert result.answers() == {"queue": "billing", "urgent": "false", "needs_human": "true"}
    # Nothing is generated, so there is nothing to bill for output tokens.
    assert result.output_tokens == 0
    assert result.cost_usd == 0.0


def test_openai_backend_prices_the_call_from_reported_usage() -> None:
    payload = json.dumps(
        {
            "queue": "technical",
            "queue_confidence": 0.88,
            "urgent": True,
            "urgent_confidence": 0.91,
            "needs_human": False,
            "needs_human_confidence": 0.6,
        }
    )
    client = FakeOpenAIClient(payload, prompt_tokens=1_000_000, completion_tokens=1_000_000)

    result = OpenAIBackend(model="gpt-4.1-nano", client=client).classify("500s everywhere")

    assert result.answers() == {"queue": "technical", "urgent": "true", "needs_human": "false"}
    # One million of each token at $0.10 in / $0.40 out.
    assert result.cost_usd == pytest.approx(0.50)
    assert client.completions.calls[0]["response_format"]["json_schema"]["strict"] is True


def test_openai_schema_offers_exactly_the_queues_laya_scores() -> None:
    client = FakeOpenAIClient(
        json.dumps(
            {
                "queue": "other",
                "queue_confidence": 0.5,
                "urgent": False,
                "urgent_confidence": 0.5,
                "needs_human": False,
                "needs_human_confidence": 0.5,
            }
        )
    )

    OpenAIBackend(model="gpt-4.1-nano", client=client).classify("hello")

    schema = client.completions.calls[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["queue"]["enum"] == list(QUEUES)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.9, 0.9), (95, 0.95), (1.0, 1.0), (-2, 0.0), (250, 1.0), ("nonsense", 0.5), (None, 0.5)],
)
def test_self_reported_confidence_is_clamped_into_a_probability(raw: object, expected: float) -> None:
    assert _clamp_confidence(raw) == pytest.approx(expected)


def test_price_lookup_accepts_dated_model_snapshots() -> None:
    assert price_for("gpt-4.1-nano-2025-04-14") == price_for("gpt-4.1-nano")
    with pytest.raises(KeyError):
        price_for("some-other-vendor-model")
