"""The evaluation runner keeps evidence; the label audit must not measure anchoring."""

import json
from pathlib import Path

from laya_router.data import Request, read_jsonl
from laya_router.evaluate import agreement, label_blind, run
from tests.fakes import FakeBackend, FakeOpenAIClient

REQUESTS = [
    Request(
        id="a",
        message="Convert 10 miles to km.",
        difficulty="clear",
        labels={"tier": "small", "needs_tools": "false", "is_sensitive": "false"},
    ),
    Request(
        id="b",
        message="Design our sharding strategy.",
        difficulty="clear",
        labels={"tier": "powerful", "needs_tools": "false", "is_sensitive": "false"},
    ),
]


def test_run_keeps_the_gold_labels_beside_the_prediction(tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"

    rows = run(FakeBackend(), REQUESTS, out_path=out)

    assert [row["id"] for row in rows] == ["a", "b"]
    assert rows[0]["tier_gold"] == "small"
    assert rows[0]["tier_answer"] == "medium"
    # Written to disk in the same shape, so a run can be re-scored without re-running it.
    assert list(read_jsonl(out)) == rows


def test_blind_labelling_never_shows_the_model_the_gold_labels() -> None:
    """The whole point of the design: an anchored judge agrees with whatever label it is shown.

    So the only thing the model may receive about a request is the request itself.
    """
    client = FakeOpenAIClient(json.dumps({"tier": "small", "needs_tools": False, "is_sensitive": False}))

    rows = label_blind(client, REQUESTS, model="gpt-5")

    assert len(client.completions.calls) == len(REQUESTS)
    for call, request in zip(client.completions.calls, REQUESTS, strict=True):
        roles = [message["role"] for message in call["messages"]]
        assert roles == ["system", "user"]
        # Exactly the request text, with nothing appended about what the answer should be.
        assert call["messages"][1]["content"] == request.message
        assert "proposed" not in call["messages"][0]["content"].lower()

    assert rows[0]["tier_mine"] == "small"
    assert rows[0]["tier_theirs"] == "small"
    assert rows[0]["agree"] is True
    # The second request is labelled `powerful` but the stub always answers `small`.
    assert rows[1]["agree"] is False


def test_agreement_is_per_question_plus_all_three() -> None:
    rows = [
        {
            "tier_mine": "small",
            "tier_theirs": "small",
            "needs_tools_mine": "false",
            "needs_tools_theirs": "true",
            "is_sensitive_mine": "false",
            "is_sensitive_theirs": "false",
            "agree": False,
        },
        {
            "tier_mine": "powerful",
            "tier_theirs": "medium",
            "needs_tools_mine": "true",
            "needs_tools_theirs": "true",
            "is_sensitive_mine": "true",
            "is_sensitive_theirs": "true",
            "agree": False,
        },
    ]

    scores = agreement(rows)

    assert scores["tier"] == 0.5
    assert scores["needs_tools"] == 0.5
    assert scores["is_sensitive"] == 1.0
    assert scores["all_three"] == 0.0
