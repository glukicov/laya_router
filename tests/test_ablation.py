"""The ablation must change only the queue wording, and must never score against absent options."""

from typing import Any

from laya_router.ablation import NO_CATCH_ALL, VARIANTS, questions_for, run
from laya_router.backends.base import BackendName
from laya_router.data import Request
from laya_router.questions import QUESTIONS


def test_only_the_queue_criteria_change() -> None:
    swapped = questions_for({"billing": "money", "other": "anything else"})

    assert swapped["queue"]["criteria"] == {"billing": "money", "other": "anything else"}
    assert swapped["queue"]["instructions"] == QUESTIONS["queue"]["instructions"]
    assert swapped["urgent"] == QUESTIONS["urgent"]
    # The shipped question set must not be mutated by building a variant.
    assert QUESTIONS["queue"]["criteria"] != swapped["queue"]["criteria"]


def test_variant_without_a_catch_all_drops_messages_it_cannot_answer() -> None:
    class StubAgent:
        def __init__(self) -> None:
            self.seen_options: list[list[str]] = []

        def predict(self, state: dict[str, str], questions: dict[str, Any]) -> dict[str, Any]:
            del state
            options = list(questions["queue"]["criteria"])
            self.seen_options.append(options)
            return {"answers": {"queue": {"type": "choice", "choice": options[0], "confidence": 0.5}}}

    class StubBackend:
        name: BackendName = "laya"
        model = "stub"

        def __init__(self, agent: StubAgent) -> None:
            self.agent = agent

        def warmup(self) -> float:
            return 0.0

        def classify(self, message: str) -> Any:
            raise AssertionError("the ablation must go through the agent, not classify()")

    agent = StubAgent()
    requests = [
        Request(
            id="a",
            message="m",
            difficulty="clear",
            labels={"queue": "billing", "urgent": "false", "needs_human": "false"},
        ),
        Request(
            id="b",
            message="m",
            difficulty="clear",
            labels={"queue": "other", "urgent": "false", "needs_human": "false"},
        ),
    ]

    rows = run(StubBackend(agent), requests, variants={"no-catch-all": NO_CATCH_ALL})

    # The `other` message has no correct option to pick, so it is excluded rather than counted as a failure.
    assert rows[0]["n"] == 1
    assert "other" not in agent.seen_options[0]


def test_every_shipped_variant_offers_at_least_five_queues() -> None:
    for name, criteria in VARIANTS.items():
        assert len(criteria) >= 5, name
        assert all(isinstance(text, str) and text for text in criteria.values()), name
