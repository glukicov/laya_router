"""The labelled set is the experiment. If it is wrong, every number downstream is wrong."""

import json
from pathlib import Path

import pytest

from laya_router.data import DATASET, load_requests
from laya_router.questions import BOOLEAN_IDS, TIERS

BASE_ROW = {
    "id": "x1",
    "message": "hi",
    "tier": "small",
    "needs_tools": "false",
    "is_sensitive": "false",
    "difficulty": "clear",
}


def test_shipped_dataset_is_valid_and_balanced() -> None:
    requests = load_requests()

    assert len(requests) >= 100
    tiers = [r.labels["tier"] for r in requests]
    assert set(tiers) == set(TIERS), "every tier must appear"
    # A router evaluated on a set that is 80% one tier can score well by always guessing it.
    for tier in TIERS:
        assert 0.2 < tiers.count(tier) / len(tiers) < 0.5, f"{tier} is over- or under-represented"

    for qid in BOOLEAN_IDS:
        values = [r.labels[qid] for r in requests]
        assert 0.1 < values.count("true") / len(values) < 0.9, f"{qid} is too one-sided to be informative"


def test_sensitivity_is_not_just_a_synonym_for_the_top_tier() -> None:
    """The interesting routing failure is confusing a risky *topic* with a hard *task*.

    If every sensitive request were also `powerful`, a router could score perfectly on both questions with one
    rule, and the evaluation would not test the distinction it claims to.
    """
    requests = load_requests()
    sensitive = [r for r in requests if r.labels["is_sensitive"] == "true"]

    assert len(sensitive) >= 20
    not_powerful = [r for r in sensitive if r.labels["tier"] != "powerful"]
    assert len(not_powerful) >= 5, "need sensitive requests that are still cheap to answer"


def test_load_rejects_an_unknown_tier(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(BASE_ROW | {"tier": "enormous"}) + "\n")

    with pytest.raises(ValueError, match="enormous"):
        load_requests(path)


def test_load_rejects_boolean_labels_that_are_not_strings(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(BASE_ROW | {"needs_tools": True}) + "\n")

    with pytest.raises(ValueError, match="needs_tools"):
        load_requests(path)


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dupes.jsonl"
    path.write_text((json.dumps(BASE_ROW) + "\n") * 2)

    with pytest.raises(ValueError, match="duplicate request ids"):
        load_requests(path)


def test_limit_takes_a_prefix() -> None:
    assert [r.id for r in load_requests(DATASET, limit=3)] == [r.id for r in load_requests()][:3]
