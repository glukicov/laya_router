"""Loading and validating the labelled evaluation set."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from laya_router.questions import BOOLEAN_IDS, QUEUES

DATASET = Path(__file__).resolve().parents[2] / "data" / "requests.jsonl"

#: How each item was written, so results can be broken down by how hard the message is rather than
#: reported as one average that hides where a model actually struggles.
DIFFICULTIES = ("clear", "terse", "negated", "mixed", "noisy")


@dataclass(frozen=True, slots=True)
class Request:
    """One labelled support message."""

    id: str
    message: str
    difficulty: str
    labels: dict[str, str]


def _validate(raw: dict[str, Any], line_number: int) -> Request:
    """Turn one JSONL row into a `Request`, failing loudly on a bad label.

    A typo in a gold label is invisible in the metrics and quietly wrong in the write-up, so the
    dataset is checked on every load rather than trusted.
    """
    missing = {"id", "message", "queue", "difficulty", *BOOLEAN_IDS} - set(raw)
    if missing:
        raise ValueError(f"line {line_number}: missing {sorted(missing)}")
    if raw["queue"] not in QUEUES:
        raise ValueError(f"line {line_number}: queue {raw['queue']!r} is not one of {sorted(QUEUES)}")
    if raw["difficulty"] not in DIFFICULTIES:
        raise ValueError(f"line {line_number}: difficulty {raw['difficulty']!r} is not one of {list(DIFFICULTIES)}")
    for qid in BOOLEAN_IDS:
        if raw[qid] not in ("true", "false"):
            raise ValueError(f"line {line_number}: {qid} must be the string 'true' or 'false', got {raw[qid]!r}")
    if not str(raw["message"]).strip():
        raise ValueError(f"line {line_number}: message is empty")

    labels = {"queue": str(raw["queue"])} | {qid: str(raw[qid]) for qid in BOOLEAN_IDS}
    return Request(id=str(raw["id"]), message=str(raw["message"]), difficulty=str(raw["difficulty"]), labels=labels)


def load_requests(path: Path = DATASET, limit: int | None = None) -> list[Request]:
    """Read the labelled set, checking every row and rejecting duplicate ids."""
    requests: list[Request] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            requests.append(_validate(json.loads(line), line_number))

    ids = [request.id for request in requests]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate request ids: {duplicates}")
    return requests[:limit] if limit else requests


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream any JSONL file this project writes."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows as JSONL, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path
