"""One ablation: does the wording of the option descriptions move the answers?

The first Laya run put most `sales` and `account` messages into `other`, which is the option described as
"none of the above fits". That is a hypothesis about the schema, not about the model, and it is cheap to test:
keep the model, the messages and the labels fixed, change only how the six queues are described, and re-score.

The variants are scored on the same 150 messages the hypothesis came from, so this measures how sensitive the
answers are to wording. It does not establish that the sharper wording generalises.
"""

from collections.abc import Sequence
from typing import Any

from laya_router.backends.base import Backend
from laya_router.data import Request
from laya_router.metrics import macro_f1
from laya_router.questions import QUESTIONS

#: The shipped descriptions, restated so the variants can be read side by side.
BASELINE = QUESTIONS["queue"]["criteria"]

#: Every queue gets concrete surface forms, and `other` stops being an open invitation.
SHARPENED = {
    "billing": "invoices, charges, refunds, plans, payment methods, tax, receipts, purchase orders",
    "technical": "bugs, outages, errors, integrations, API problems, SDKs, performance, documentation",
    "sales": "pricing, quotes, demos, trials, upgrades, renewals, contracts, procurement, partnerships",
    "account": "login, passwords, two-factor, SSO, permissions, seats, user management, profile and data requests",
    "abuse": "spam, phishing, fraud, harassment, impersonation, or a compromised or misused account",
    "other": "only when the message is about none of these: press, careers, events, feedback, or a wrong recipient",
}

#: Removing the catch-all entirely: five real queues and nothing to fall back on.
NO_CATCH_ALL = {name: text for name, text in SHARPENED.items() if name != "other"}

VARIANTS: dict[str, dict[str, str]] = {
    "shipped": dict(BASELINE),
    "sharpened": SHARPENED,
    "no-catch-all": NO_CATCH_ALL,
}


def questions_for(criteria: dict[str, str]) -> dict[str, Any]:
    """The shipped question set with only the queue descriptions swapped."""
    questions = {qid: dict(q) for qid, q in QUESTIONS.items()}
    questions["queue"] = {**questions["queue"], "criteria": criteria}
    return questions


def run(
    backend: Backend,
    requests: Sequence[Request],
    variants: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Re-answer the queue question under each wording and score it.

    Only the queue question is asked, because only its options change; the yes/no questions are untouched and
    re-running them would just burn time for identical answers.
    """
    agent = getattr(backend, "agent", None)
    if agent is None:
        raise TypeError("the ablation needs a backend with a resident Laya agent")

    chosen = variants or VARIANTS
    # Removing an option also removes the messages it was the right answer for, and those are not a random
    # sample: they are exactly the hard ones. Scoring each variant on its own subset would compare a 5-way
    # problem on 130 easy messages with a 6-way problem on 150, and call the difference an improvement. So
    # every variant is also scored on the messages every variant can answer.
    common_queues = set.intersection(*(set(criteria) for criteria in chosen.values()))
    common = [r for r in requests if r.labels["queue"] in common_queues]

    rows: list[dict[str, Any]] = []
    for name, criteria in chosen.items():
        questions = {"queue": questions_for(criteria)["queue"]}
        scorable = [r for r in requests if r.labels["queue"] in criteria]
        predicted = {
            r.id: str(agent.predict({"message": r.message}, questions)["answers"]["queue"]["choice"]) for r in scorable
        }
        gold = {r.id: r.labels["queue"] for r in scorable}
        common_ids = [r.id for r in common]
        rows.append(
            {
                "variant": name,
                "options": len(criteria),
                "n": len(scorable),
                "accuracy": sum(predicted[i] == gold[i] for i in predicted) / len(predicted),
                "macro_f1": macro_f1(list(predicted.values()), list(gold.values())),
                "other_rate": list(predicted.values()).count("other") / len(predicted),
                "n_common": len(common_ids),
                "accuracy_common": sum(predicted[i] == gold[i] for i in common_ids) / len(common_ids),
                "macro_f1_common": macro_f1([predicted[i] for i in common_ids], [gold[i] for i in common_ids]),
            }
        )
    return rows
