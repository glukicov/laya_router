"""The one routing schema both backends answer.

A smart router sits in front of a model fleet and decides, per request, which tier should answer it: a small
model for routine work, a mid-size one for general analysis, a frontier one for genuinely hard or high-stakes
requests. The router runs on *every* request, so its own latency and cost are pure overhead — which is exactly
what makes it an interesting thing to put a 421M encoder behind.

Both backends answer these same questions. Laya consumes `QUESTIONS` directly (it is the SDK's native typed-
question format); the OpenAI backend renders the same dictionary into a prompt and a JSON schema, so neither
side is given wording the other did not.
"""

from typing import Any, Final

TIERS: Final[dict[str, str]] = {
    "small": "routine and predictable: a lookup, a greeting, a format change, a short rewrite, a one-line answer",
    "medium": "general analysis: several steps, ordinary code, a summary that needs judgement, a routine explanation",
    "powerful": (
        "complex or high-risk: long multi-step reasoning, specialist knowledge, system design, "
        "or consequences in money, law, health or safety"
    ),
}

QUESTIONS: Final[dict[str, dict[str, Any]]] = {
    "tier": {
        "type": "choice",
        "instructions": (
            "Which model tier should answer the user's `request`? Pick the cheapest tier that can do it well."
        ),
        "criteria": TIERS,
    },
    "needs_tools": {
        "type": "noul",
        "instructions": (
            "Does answering `request` need information the request does not already contain, such as a live "
            "system, private records, or a current fact? Answer false when the material to work on arrives "
            "with the request."
        ),
    },
    "is_sensitive": {
        "type": "noul",
        "instructions": (
            "Does `request` carry real-world consequences in money, law, health or safety, such that a "
            "wrong answer would cause harm?"
        ),
    },
}

#: Question ids in the order used by every report and figure.
QUESTION_IDS: Final[tuple[str, ...]] = tuple(QUESTIONS)

#: Question ids whose answer is a yes/no (Laya's `noul` type).
BOOLEAN_IDS: Final[tuple[str, ...]] = ("needs_tools", "is_sensitive")

#: The routing decision itself: the one answer that picks which model runs next.
ROUTE_ID: Final[str] = "tier"

#: Tiers from cheapest to most capable, so a wrong route can be scored by how far off it was.
TIER_ORDER: Final[tuple[str, ...]] = ("small", "medium", "powerful")


def render_for_prompt() -> str:
    """Render `QUESTIONS` as the instruction block given to a generative classifier.

    The OpenAI backend cannot read Laya's typed-question dictionary, so it gets this rendering of the exact
    same content. Keeping the rendering here, rather than in a hand-written prompt string, is what stops the
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
