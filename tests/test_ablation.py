"""The ablation must change only the tier wording, and must score the route by direction of error."""

from typing import Any

from laya_router.ablation import BARE, VARIANTS, questions_for, run
from laya_router.backends.base import BackendName
from laya_router.data import Request
from laya_router.metrics import routing_errors
from laya_router.questions import QUESTIONS


def test_only_the_tier_criteria_change() -> None:
    swapped = questions_for({"small": "cheap", "medium": "middling", "powerful": "dear"})

    assert swapped["tier"]["criteria"] == {"small": "cheap", "medium": "middling", "powerful": "dear"}
    assert swapped["tier"]["instructions"] == QUESTIONS["tier"]["instructions"]
    assert swapped["needs_tools"] == QUESTIONS["needs_tools"]
    # Building a variant must not mutate the shipped question set.
    assert QUESTIONS["tier"]["criteria"] != swapped["tier"]["criteria"]


def test_every_variant_offers_all_three_tiers() -> None:
    for name, criteria in VARIANTS.items():
        assert list(criteria) == ["small", "medium", "powerful"], name


def test_run_asks_only_the_tier_question_and_scores_the_direction_of_error() -> None:
    class StubAgent:
        def __init__(self) -> None:
            self.asked: list[list[str]] = []

        def predict(self, state: dict[str, str], questions: dict[str, Any]) -> dict[str, Any]:
            del state
            self.asked.append(sorted(questions))
            # Always routes to the cheapest tier, so every non-small request is underspent.
            return {"answers": {"tier": {"type": "choice", "choice": "small", "confidence": 0.5}}}

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
            labels={"tier": "small", "needs_tools": "false", "is_sensitive": "false"},
        ),
        Request(
            id="b",
            message="m",
            difficulty="clear",
            labels={"tier": "powerful", "needs_tools": "false", "is_sensitive": "false"},
        ),
    ]

    rows = run(StubBackend(agent), requests, variants={"names only": BARE})

    assert agent.asked == [["tier"], ["tier"]], "the yes/no questions must not be re-asked"
    assert rows[0]["accuracy"] == 0.5
    assert rows[0]["underspend_rate"] == 0.5
    assert rows[0]["overspend_rate"] == 0.0
    assert rows[0]["share"] == {"small": 1.0, "medium": 0.0, "powerful": 0.0}


def test_routing_errors_separates_overspend_from_underspend() -> None:
    gold = ["small", "small", "powerful", "medium"]
    predicted = ["powerful", "small", "small", "medium"]

    errors = routing_errors(predicted, gold)

    # One request sent two tiers too high, one sent two tiers too low, two correct.
    assert errors["overspend_rate"] == 0.25
    assert errors["underspend_rate"] == 0.25
    assert errors["two_tiers_off_rate"] == 0.5
