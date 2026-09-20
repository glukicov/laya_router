"""The labelled set is the experiment. If it is wrong, every number downstream is wrong."""

import json
from pathlib import Path

import pytest

from laya_router.data import DATASET, load_requests
from laya_router.questions import BOOLEAN_IDS, QUEUES


def test_shipped_dataset_is_valid_and_balanced() -> None:
    requests = load_requests()

    assert len(requests) >= 100
    assert {r.labels["queue"] for r in requests} == set(QUEUES), "every queue must appear"
    for qid in BOOLEAN_IDS:
        values = [r.labels[qid] for r in requests]
        # A boolean question answered 'false' everywhere would be trivially solved by a constant.
        assert 0.1 < values.count("true") / len(values) < 0.9, f"{qid} is too one-sided to be informative"


def test_load_rejects_an_unknown_queue(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    row = {
        "id": "x1",
        "message": "hi",
        "queue": "marketing",
        "urgent": "false",
        "needs_human": "false",
        "difficulty": "clear",
    }
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="marketing"):
        load_requests(path)


def test_load_rejects_boolean_labels_that_are_not_strings(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    row = {
        "id": "x1",
        "message": "hi",
        "queue": "billing",
        "urgent": True,
        "needs_human": "false",
        "difficulty": "clear",
    }
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="urgent"):
        load_requests(path)


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dupes.jsonl"
    row = {
        "id": "x1",
        "message": "hi",
        "queue": "billing",
        "urgent": "false",
        "needs_human": "false",
        "difficulty": "clear",
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="duplicate request ids"):
        load_requests(path)


def test_limit_takes_a_prefix() -> None:
    assert [r.id for r in load_requests(DATASET, limit=3)] == [r.id for r in load_requests()][:3]
