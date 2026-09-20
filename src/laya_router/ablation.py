"""One ablation: how much of the routing decision comes from the wording of the tier descriptions?

A router's options are not fixed facts, they are three sentences someone wrote. If rewording them moves the
route substantially, then any accuracy number is partly a number about that prose — worth knowing before
trusting either backend's score.

Model, requests and labels stay fixed; only the three tier descriptions change, and the answers are re-scored.
The variants are evaluated on the same messages that generated the hypothesis, so this measures sensitivity to
wording. It does not establish that any variant generalises.
"""

from collections.abc import Sequence
from typing import Any

from laya_router.backends.base import Backend
from laya_router.data import Request
from laya_router.metrics import macro_f1
from laya_router.questions import QUESTIONS, TIER_ORDER

#: The shipped descriptions, named so the variants can be read side by side.
BASELINE: dict[str, str] = dict(QUESTIONS["tier"]["criteria"])

#: Names only. Whatever the model already associates with small, medium and powerful, and nothing else.
BARE = {"small": "", "medium": "", "powerful": ""}

#: Concrete requests instead of abstract categories.
EXAMPLE_LED = {
    "small": "like: convert these units, fix this typo, what is the capital of Peru, reformat this list",
    "medium": "like: write this function, summarise this document, explain this error, draft this email",
    "powerful": "like: design this system, is this contract enforceable, prove this, plan this migration",
}

#: Cost framed as the decision, rather than capability. A router's actual job is to not overspend.
COST_FRAMED = {
    "small": "cheapest: use it whenever it would plausibly be enough",
    "medium": "roughly ten times the cost of small: use it when small would clearly fail",
    "powerful": "roughly a hundred times the cost of small: use it only when being wrong would be expensive",
}

VARIANTS: dict[str, dict[str, str]] = {
    "shipped": BASELINE,
    "names only": BARE,
    "example-led": EXAMPLE_LED,
    "cost-framed": COST_FRAMED,
}


def questions_for(criteria: dict[str, str]) -> dict[str, Any]:
    """The shipped question set with only the tier descriptions swapped."""
    questions = {qid: dict(q) for qid, q in QUESTIONS.items()}
    questions["tier"] = {**questions["tier"], "criteria": criteria}
    return questions


def run(
    backend: Backend,
    requests: Sequence[Request],
    variants: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Re-answer the tier question under each wording and score it.

    Only the tier question is asked: it is the only one whose options change, and re-running the two yes/no
    questions would burn time for identical answers.
    """
    agent = getattr(backend, "agent", None)
    if agent is None:
        raise TypeError("the ablation needs a backend with a resident Laya agent")

    rows: list[dict[str, Any]] = []
    for name, criteria in (variants or VARIANTS).items():
        questions = {"tier": questions_for(criteria)["tier"]}
        predicted = [
            str(agent.predict({"message": r.message}, questions)["answers"]["tier"]["choice"]) for r in requests
        ]
        gold = [r.labels["tier"] for r in requests]
        rows.append(
            {
                "variant": name,
                "n": len(requests),
                "accuracy": sum(p == g for p, g in zip(predicted, gold, strict=True)) / len(gold),
                "macro_f1": macro_f1(predicted, gold),
                "overspend_rate": _overspend(predicted, gold),
                "underspend_rate": _underspend(predicted, gold),
                "share": {tier: predicted.count(tier) / len(predicted) for tier in TIER_ORDER},
            }
        )
    return rows


def _overspend(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Share of requests sent to a more expensive tier than they needed. This costs money."""
    return sum(TIER_ORDER.index(p) > TIER_ORDER.index(g) for p, g in zip(predicted, gold, strict=True)) / len(gold)


def _underspend(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Share of requests sent to a tier too weak for them. This costs answer quality."""
    return sum(TIER_ORDER.index(p) < TIER_ORDER.index(g) for p, g in zip(predicted, gold, strict=True)) / len(gold)
