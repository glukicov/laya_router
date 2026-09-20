"""The traditional backend: a hosted LLM asked to classify, with structured outputs.

This is the version almost everyone builds first, so it is the one worth measuring against. The
model is given the same questions as Laya and a strict JSON schema to fill in, which removes the
usual parsing problems but not the two costs that matter: a network round trip per message, and a
bill per token.
"""

import json
import os
from time import perf_counter
from typing import Any

from laya_router.backends.base import BackendName
from laya_router.questions import BOOLEAN_IDS, TIERS, render_for_prompt
from laya_router.schema import Decision, RouteResult

DEFAULT_MODEL = "gpt-5-nano"

# USD per million tokens (input, output), read from OpenAI's pricing page in February 2026. Prices
# move; `laya-router eval run --price-in/--price-out` overrides these without editing code, and
# docs/EVAL.md states which numbers produced the published figures.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}

SYSTEM_PROMPT = (
    "You are the router in front of a fleet of language models. For each incoming user request, answer "
    "every question below using only what the request says. For each question also give a confidence "
    "between 0 and 1: the probability that your own answer is correct.\n\n"
    f"{render_for_prompt()}"
)


def _response_schema() -> dict[str, Any]:
    """Build the strict JSON schema from the shared question set.

    Deriving this from `QUESTIONS` rather than writing it out by hand is what guarantees the hosted
    model is asked for exactly the options Laya scores, including their order.
    """
    properties: dict[str, Any] = {
        "tier": {"type": "string", "enum": list(TIERS)},
        "tier_confidence": {"type": "number"},
    }
    for qid in BOOLEAN_IDS:
        properties[qid] = {"type": "boolean"}
        properties[f"{qid}_confidence"] = {"type": "number"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def price_for(model: str) -> tuple[float, float]:
    """USD per million input and output tokens for `model`, matching on the dated-snapshot suffix."""
    if model in PRICES:
        return PRICES[model]
    for known, prices in PRICES.items():
        if model.startswith(known):
            return prices
    raise KeyError(f"no price known for {model!r}; pass --price-in and --price-out explicitly")


class OpenAIBackend:
    """Route one request per API call, with the answer constrained by a JSON schema."""

    name: BackendName = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        price_in: float | None = None,
        price_out: float | None = None,
        client: Any | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        default_in, default_out = price_for(model)
        self.price_in = default_in if price_in is None else price_in
        self.price_out = default_out if price_out is None else price_out
        self._schema = _response_schema()
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set; copy .env.example to .env and fill it in")
            self.client = OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)

    def warmup(self) -> float:
        """Open the connection before timing anything, so TLS setup is not counted as latency."""
        started = perf_counter()
        self.classify("Convert 10 miles to kilometres.")
        return perf_counter() - started

    def classify(self, message: str) -> RouteResult:
        """Send one message and parse the schema-constrained reply."""
        started = perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "route", "strict": True, "schema": self._schema},
            },
        )
        latency_ms = (perf_counter() - started) * 1_000

        payload = json.loads(response.choices[0].message.content or "{}")
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        decisions = {"tier": _string_decision(payload, "tier")}
        for qid in BOOLEAN_IDS:
            decisions[qid] = _boolean_decision(payload, qid)

        return RouteResult(
            backend="openai",
            model=self.model,
            decisions=decisions,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=(input_tokens * self.price_in + output_tokens * self.price_out) / 1_000_000,
        )


def _clamp_confidence(value: Any) -> float:
    """Keep a self-reported confidence inside [0, 1].

    The schema asks for a probability but only constrains the type, and models do occasionally
    answer 95 when they mean 0.95. Clamping keeps one malformed number from poisoning the
    calibration analysis; the raw value stays in the saved response.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if number > 1.0:
        number = number / 100.0 if number <= 100.0 else 1.0
    return min(max(number, 0.0), 1.0)


def _string_decision(payload: dict[str, Any], qid: str) -> Decision:
    """Read one enum answer. A generative classifier reports no distribution, only a point answer."""
    answer = str(payload.get(qid, "other"))
    confidence = _clamp_confidence(payload.get(f"{qid}_confidence", 0.5))
    return Decision(answer=answer, confidence=confidence, probabilities={answer: confidence})


def _boolean_decision(payload: dict[str, Any], qid: str) -> Decision:
    """Read one yes/no answer, normalised to the same 'true'/'false' strings Laya produces."""
    answer = "true" if bool(payload.get(qid, False)) else "false"
    confidence = _clamp_confidence(payload.get(f"{qid}_confidence", 0.5))
    return Decision(
        answer=answer,
        confidence=confidence,
        probabilities={answer: confidence, ("false" if answer == "true" else "true"): 1.0 - confidence},
    )
