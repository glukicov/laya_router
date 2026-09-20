"""Run a backend over the labelled set and record what happened.

One row per message per backend, written as JSONL so a run can be inspected, diffed and re-scored
without re-running anything. Scoring lives in `metrics`; this module only produces evidence.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from laya_router.backends.base import Backend
from laya_router.data import Request, write_jsonl
from laya_router.questions import QUESTION_IDS

RESULTS = Path(__file__).resolve().parents[2] / "results"


def run(
    backend: Backend,
    requests: Sequence[Request],
    out_path: Path | None = None,
    on_result: Callable[[int, Request, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Classify every request once, keeping the gold labels alongside the prediction.

    Requests are sent one at a time, in order. That is not the fastest way to use a hosted API, but
    it is the only way the recorded latency means anything: a concurrent run would measure the
    client's pipelining rather than the service's response time, and the two backends would no
    longer be comparable.
    """
    rows: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        result = backend.classify(request.message)
        row = {
            "id": request.id,
            "difficulty": request.difficulty,
            **result.to_row(),
            **{f"{qid}_gold": request.labels[qid] for qid in QUESTION_IDS},
        }
        rows.append(row)
        if on_result is not None:
            on_result(index, request, row)

    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows


BLIND_PROMPT = (
    "You are labelling a dataset of user requests for a model router. For the request below, answer every "
    "question from scratch. You are not shown anyone else's answers; give your own."
)


def label_blind(
    client: Any, requests: Sequence[Request], model: str, out_path: Path | None = None
) -> list[dict[str, Any]]:
    """Have a strong, independent model label the set from scratch, never seeing the gold labels.

    This replaces an earlier design that showed the model the proposed labels and asked whether it agreed.
    That design does not measure what it looks like it measures: run it twice, either side of a relabelling,
    and the same model objects to whichever label it is shown, in whichever direction. Anchoring on the
    presented answer makes the resulting "agreement rate" meaningless.

    Labelling blind and comparing afterwards is the ordinary way to measure inter-annotator agreement, and it
    is the number this project reports.
    """
    from laya_router.questions import QUESTIONS, TIERS, render_for_prompt

    schema = {
        "type": "object",
        "properties": {
            "tier": {"type": "string", "enum": list(TIERS)},
            **{qid: {"type": "boolean"} for qid in QUESTIONS if qid != "tier"},
        },
        "required": list(QUESTIONS),
        "additionalProperties": False,
    }

    rows: list[dict[str, Any]] = []
    for request in requests:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{BLIND_PROMPT}\n\n{render_for_prompt()}"},
                {"role": "user", "content": request.message},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "labels", "strict": True, "schema": schema},
            },
        )
        proposed = json.loads(response.choices[0].message.content or "{}")
        theirs = {
            qid: (str(proposed[qid]).lower() if isinstance(proposed[qid], bool) else str(proposed[qid]))
            for qid in QUESTIONS
            if qid in proposed
        }
        rows.append(
            {
                "id": request.id,
                "message": request.message,
                **{f"{qid}_mine": value for qid, value in request.labels.items()},
                **{f"{qid}_theirs": value for qid, value in theirs.items()},
                "agree": all(theirs.get(qid) == value for qid, value in request.labels.items()),
            }
        )

    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows


def agreement(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Per-question agreement between the gold labels and an independent blind pass."""
    out: dict[str, float] = {}
    for qid in QUESTION_IDS:
        pairs = [(r[f"{qid}_mine"], r.get(f"{qid}_theirs")) for r in rows if f"{qid}_theirs" in r]
        if pairs:
            out[qid] = sum(mine == theirs for mine, theirs in pairs) / len(pairs)
    out["all_three"] = sum(bool(r["agree"]) for r in rows) / len(rows) if rows else 0.0
    return out
