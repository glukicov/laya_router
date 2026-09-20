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


ADJUDICATION_PROMPT = (
    "You are auditing the gold labels of a support-triage dataset. For the message below, decide "
    "whether the proposed labels are defensible. Answer `agree` unless a label is clearly wrong; "
    "borderline cases count as agreement. If you disagree, say which label and what it should be."
)


def adjudicate(
    client: Any, requests: Sequence[Request], model: str, out_path: Path | None = None
) -> list[dict[str, Any]]:
    """Ask a strong, independent model whether each gold label is defensible.

    The labels here were written by one person, which is a real weakness of a hand-authored set.
    This does not fix that, but it does surface the items where a capable reader would disagree, so
    the write-up can report how many there are instead of claiming the labels are obviously right.
    """
    from laya_router.questions import render_for_prompt

    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["agree", "disagree"]},
            "field": {"type": "string", "enum": [*QUESTION_IDS, "none"]},
            "suggested": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "field", "suggested", "reason"],
        "additionalProperties": False,
    }

    rows: list[dict[str, Any]] = []
    for request in requests:
        proposed = json.dumps(request.labels, ensure_ascii=False)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{ADJUDICATION_PROMPT}\n\n{render_for_prompt()}"},
                {"role": "user", "content": f"message: {request.message}\nproposed labels: {proposed}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "adjudication", "strict": True, "schema": schema},
            },
        )
        verdict = json.loads(response.choices[0].message.content or "{}")
        rows.append({"id": request.id, "message": request.message, **request.labels, **verdict})

    if out_path is not None:
        write_jsonl(out_path, rows)
    return rows
