"""Scoring for the head-to-head study.

Deliberately dependency-free: every number in docs/EVAL.md is computed here in plain Python, so the
scoring can be read end to end and runs in CI without installing the evaluation extras.
"""

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from laya_router.data import Request
from laya_router.questions import BOOLEAN_IDS, QUESTION_IDS, ROUTE_ID, TIER_ORDER

#: Bins used for expected calibration error and the reliability diagram.
CALIBRATION_BINS = 10


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, so p50 and p99 mean the same thing for both backends."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def macro_f1(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Unweighted mean F1 over the classes that appear in the gold labels.

    Macro rather than micro because the queues are unbalanced: a model that never predicts `abuse`
    should be visibly penalised, and accuracy alone would barely notice.
    """
    classes = sorted(set(gold))
    scores = []
    for label in classes:
        tp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predicted, gold, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predicted, gold, strict=True) if p != label and g == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = CALIBRATION_BINS
) -> float:
    """Average gap between stated confidence and observed accuracy, weighted by bin population.

    This is the number that says whether a confidence score can be used as a threshold. A model can
    be accurate and badly calibrated at the same time, and only the second failure makes automated
    escalation unsafe.
    """
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for lower, upper, members in _bin_members(confidences, correct, bins):
        del lower, upper
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(1 for _, ok in members if ok) / len(members)
        error += (len(members) / total) * abs(mean_confidence - accuracy)
    return error


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = CALIBRATION_BINS
) -> list[dict[str, float]]:
    """Per-bin confidence and accuracy, for the reliability diagram."""
    out = []
    for lower, upper, members in _bin_members(confidences, correct, bins):
        if not members:
            continue
        out.append(
            {
                "lower": lower,
                "upper": upper,
                "count": float(len(members)),
                "confidence": sum(c for c, _ in members) / len(members),
                "accuracy": sum(1 for _, ok in members if ok) / len(members),
            }
        )
    return out


def _bin_members(
    confidences: Sequence[float], correct: Sequence[bool], bins: int
) -> list[tuple[float, float, list[tuple[float, bool]]]]:
    """Group (confidence, correct) pairs into equal-width confidence bins."""
    grouped: list[tuple[float, float, list[tuple[float, bool]]]] = [(i / bins, (i + 1) / bins, []) for i in range(bins)]
    for confidence, ok in zip(confidences, correct, strict=True):
        index = min(bins - 1, max(0, int(confidence * bins)))
        grouped[index][2].append((confidence, ok))
    return grouped


def deferral_curve(confidences: Sequence[float], correct: Sequence[bool], steps: int = 20) -> list[dict[str, float]]:
    """Accuracy on the most confident fraction of decisions, as that fraction grows.

    This is the practical question behind calibration: if the low-confidence cases are escalated to
    a person, how clean is what is left? A useful confidence signal makes this curve fall as
    coverage rises. A useless one makes it flat.
    """
    if not confidences:
        return []
    ranked = sorted(zip(confidences, correct, strict=True), key=lambda pair: pair[0], reverse=True)
    total = len(ranked)
    curve = []
    for step in range(1, steps + 1):
        kept = max(1, round(total * step / steps))
        window = ranked[:kept]
        curve.append(
            {
                "coverage": kept / total,
                "accuracy": sum(1 for _, ok in window if ok) / kept,
                "threshold": window[-1][0],
            }
        )
    return curve


def score(rows: list[dict[str, Any]], requests: Sequence[Request]) -> dict[str, Any]:
    """Every published number for one backend's run over the labelled set.

    `rows` are the per-message results written by `evaluate.run`, keyed to `requests` by id, so a
    partial or reordered run is scored correctly rather than silently misaligned.
    """
    by_id = {row["id"]: row for row in rows}
    matched = [request for request in requests if request.id in by_id]
    if not matched:
        raise ValueError("no evaluated rows matched the labelled set")

    latencies = [float(by_id[r.id]["latency_ms"]) for r in matched]
    per_question: dict[str, Any] = {}
    all_confidences: list[float] = []
    all_correct: list[bool] = []

    for qid in QUESTION_IDS:
        predicted = [str(by_id[r.id][f"{qid}_answer"]) for r in matched]
        gold = [r.labels[qid] for r in matched]
        correct = [p == g for p, g in zip(predicted, gold, strict=True)]
        confidences = [float(by_id[r.id][f"{qid}_confidence"]) for r in matched]
        all_confidences += confidences
        all_correct += correct

        entry: dict[str, Any] = {
            "accuracy": sum(correct) / len(correct),
            "ece": expected_calibration_error(confidences, correct),
            "mean_confidence": sum(confidences) / len(confidences),
            "reliability": reliability_bins(confidences, correct),
        }
        if qid not in BOOLEAN_IDS:
            entry["macro_f1"] = macro_f1(predicted, gold)
            entry["confusion"] = _confusion(predicted, gold)
        if qid == ROUTE_ID:
            entry.update(routing_errors(predicted, gold))
        per_question[qid] = entry

    exact = [all(by_id[r.id][f"{qid}_answer"] == r.labels[qid] for qid in QUESTION_IDS) for r in matched]

    return {
        "backend": rows[0]["backend"],
        "model": rows[0]["model"],
        "n": len(matched),
        "questions": per_question,
        "exact_match": sum(exact) / len(exact),
        "overall_accuracy": sum(all_correct) / len(all_correct),
        "overall_ece": expected_calibration_error(all_confidences, all_correct),
        "deferral": deferral_curve(all_confidences, all_correct),
        "by_difficulty": _by_difficulty(matched, by_id),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p99": percentile(latencies, 0.99),
            "mean": sum(latencies) / len(latencies),
        },
        "cost_usd_total": sum(float(by_id[r.id]["cost_usd"]) for r in matched),
        "cost_usd_per_1k": sum(float(by_id[r.id]["cost_usd"]) for r in matched) / len(matched) * 1_000,
        "input_tokens": sum(int(by_id[r.id]["input_tokens"]) for r in matched),
        "output_tokens": sum(int(by_id[r.id]["output_tokens"]) for r in matched),
    }


def routing_errors(predicted: Sequence[str], gold: Sequence[str]) -> dict[str, Any]:
    """Split the route's mistakes by which way they went, because they cost different things.

    Sending a request to a tier above what it needed wastes money on every such request. Sending it below
    means the answer comes back worse, which is the failure a user notices. A single accuracy number hides
    which of the two a router prefers, and that preference is the whole design question.
    """
    over = sum(TIER_ORDER.index(p) > TIER_ORDER.index(g) for p, g in zip(predicted, gold, strict=True))
    under = sum(TIER_ORDER.index(p) < TIER_ORDER.index(g) for p, g in zip(predicted, gold, strict=True))
    # Two tiers off: the route skipped a level entirely, in either direction.
    severe = sum(abs(TIER_ORDER.index(p) - TIER_ORDER.index(g)) == 2 for p, g in zip(predicted, gold, strict=True))
    return {
        "overspend_rate": over / len(gold),
        "underspend_rate": under / len(gold),
        "two_tiers_off_rate": severe / len(gold),
        "share": {tier: predicted.count(tier) / len(predicted) for tier in TIER_ORDER},
    }


def _confusion(predicted: Sequence[str], gold: Sequence[str]) -> dict[str, dict[str, int]]:
    """Counts of gold label against predicted label, for the queue question."""
    counts: dict[str, Counter[str]] = {}
    for p, g in zip(predicted, gold, strict=True):
        counts.setdefault(g, Counter())[p] += 1
    return {g: dict(c) for g, c in sorted(counts.items())}


def _by_difficulty(requests: Sequence[Request], by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Exact-match rate and sample size per difficulty tag."""
    out: dict[str, dict[str, float]] = {}
    for request in requests:
        bucket = out.setdefault(request.difficulty, {"n": 0.0, "exact_match": 0.0})
        bucket["n"] += 1
        if all(by_id[request.id][f"{qid}_answer"] == request.labels[qid] for qid in QUESTION_IDS):
            bucket["exact_match"] += 1
    for bucket in out.values():
        bucket["exact_match"] /= bucket["n"]
    return out
