"""The one triage schema both backends answer.

Every backend in this repo answers exactly these questions about one inbound support message, so the
comparison is like-for-like: same wording, same option set, same output contract. Laya consumes
`QUESTIONS` directly (it is the SDK's native typed-question format); the OpenAI backend renders the
same dictionary into a prompt and a JSON schema, so neither side gets a hand-tuned advantage.
"""

from typing import Any, Final

QUEUES: Final[dict[str, str]] = {
    "billing": "invoices, charges, refunds, plans, payment methods",
    "technical": "bugs, outages, errors, integrations, API problems",
    "sales": "pricing, quotes, demos, upgrades, new contracts",
    "account": "login, access, permissions, seats, profile and data changes",
    "abuse": "spam, fraud, harassment, or a compromised or misused account",
    "other": "none of the above fits",
}

QUESTIONS: Final[dict[str, dict[str, Any]]] = {
    "queue": {
        "type": "choice",
        "instructions": "Which support queue should handle the customer's `message`?",
        "criteria": QUEUES,
    },
    "urgent": {
        "type": "noul",
        "instructions": (
            "Does `message` describe something already broken, blocking work, or bound by a stated deadline?"
        ),
    },
    "needs_human": {
        "type": "noul",
        "instructions": (
            "Must a human agent handle `message`, rather than an automated reply? "
            "True for money movement, legal or regulatory matters, account security, and distressed customers."
        ),
    },
}

#: Question ids in the order used by every report and figure.
QUESTION_IDS: Final[tuple[str, ...]] = tuple(QUESTIONS)

#: Question ids whose answer is a yes/no (Laya's `noul` type).
BOOLEAN_IDS: Final[tuple[str, ...]] = ("urgent", "needs_human")


def render_for_prompt() -> str:
    """Render `QUESTIONS` as the instruction block given to a generative classifier.

    The OpenAI backend cannot read Laya's typed-question dictionary, so it gets this rendering of the
    exact same content. Keeping the rendering here (rather than in a prompt string) is what stops the
    two backends from drifting apart as the schema changes.
    """
    lines = []
    for qid, q in QUESTIONS.items():
        lines.append(f"{qid}: {q['instructions']}")
        criteria = q.get("criteria")
        if isinstance(criteria, dict):
            options = "\n".join(f"  - {name}: {desc}" for name, desc in criteria.items())
            lines.append(f"  one of:\n{options}")
        else:
            lines.append("  answer true or false")
    return "\n".join(lines)
