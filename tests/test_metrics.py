"""Scoring is the part of the repo most easily wrong in a way nobody notices."""

import math

from laya_router.data import Request
from laya_router.metrics import (
    deferral_curve,
    expected_calibration_error,
    macro_f1,
    percentile,
    score,
)


def test_percentile_interpolates_between_samples() -> None:
    assert percentile([10, 20, 30, 40], 0.5) == 25
    assert percentile([10], 0.99) == 10
    assert percentile([], 0.5) == 0.0


def test_macro_f1_punishes_a_class_the_model_never_predicts() -> None:
    gold = ["a", "a", "b", "b"]
    # Always answering 'a' gets half the labels right, but never finds 'b' at all:
    # F1(a) = 2/3 because recall is perfect and precision is 0.5, F1(b) = 0, so the mean is 1/3.
    assert macro_f1(["a", "a", "a", "a"], gold) == pytest_approx(1 / 3, 1e-9)
    assert macro_f1(gold, gold) == 1.0


def test_expected_calibration_error_is_zero_when_confidence_matches_outcomes() -> None:
    # Ten decisions at 0.9 confidence, nine of them correct: exactly what 0.9 should mean.
    confidences = [0.9] * 10
    correct = [True] * 9 + [False]
    assert expected_calibration_error(confidences, correct) == pytest_approx(0.0, 0.001)

    # The same accuracy claimed at certainty is a 0.1 gap.
    assert expected_calibration_error([1.0] * 10, correct) == pytest_approx(0.1, 0.001)


def test_deferral_curve_ranks_the_confident_first() -> None:
    # The two wrong answers are the two least confident ones, so accuracy falls as coverage grows.
    confidences = [0.99, 0.95, 0.60, 0.55]
    correct = [True, True, False, False]
    curve = deferral_curve(confidences, correct, steps=4)

    assert [point["coverage"] for point in curve] == [0.25, 0.5, 0.75, 1.0]
    assert curve[1]["accuracy"] == 1.0
    assert curve[-1]["accuracy"] == 0.5


def test_score_aligns_rows_to_requests_by_id_not_order() -> None:
    requests = [
        Request(
            id="a",
            message="m",
            difficulty="clear",
            labels={"tier": "small", "needs_tools": "true", "is_sensitive": "false"},
        ),
        Request(
            id="b",
            message="m",
            difficulty="terse",
            labels={"tier": "powerful", "needs_tools": "false", "is_sensitive": "false"},
        ),
    ]
    rows = [_row("b", "powerful", "false", "false"), _row("a", "small", "true", "false")]

    summary = score(rows, requests)

    assert summary["n"] == 2
    assert summary["exact_match"] == 1.0
    assert summary["by_difficulty"]["terse"]["n"] == 1
    assert summary["cost_usd_per_1k"] == 1.0


def _row(request_id: str, tier: str, needs_tools: str, is_sensitive: str) -> dict[str, object]:
    return {
        "id": request_id,
        "backend": "laya",
        "model": "fake",
        "latency_ms": 10.0,
        "input_tokens": 1,
        "output_tokens": 0,
        "cost_usd": 0.001,
        "tier_answer": tier,
        "tier_confidence": 0.9,
        "needs_tools_answer": needs_tools,
        "needs_tools_confidence": 0.9,
        "is_sensitive_answer": is_sensitive,
        "is_sensitive_confidence": 0.9,
    }


def pytest_approx(value: float, tolerance: float) -> object:
    """Tiny local approx so the assertions read as maths rather than as pytest plumbing."""

    class _Approx:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, float) and math.isclose(other, value, abs_tol=tolerance)

        def __repr__(self) -> str:
            return f"approx({value} +- {tolerance})"

    return _Approx()
