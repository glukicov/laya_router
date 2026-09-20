"""Figures for docs/EVAL.md, rendered from results/metrics.json.

The encoding is fixed across every figure so the set reads as one system: colour identifies the
backend (Laya blue, the hosted classifier orange) and nothing else, and every series is named in a
legend so identity never rests on colour alone.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from laya_router.questions import QUESTION_IDS, ROUTE_ID, TIER_ORDER

FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c9c8c2"
BACKEND_COLOUR = {"laya": BLUE, "openai": ORANGE}
BACKEND_LABEL = {"laya": "Laya 421M, local", "openai": "GPT-5 nano router"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "font.size": 10,
    }
)


def _save(fig: Figure, name: str) -> Path:
    """Write one figure into docs/figures and close it."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _label(name: str, summary: dict[str, Any]) -> str:
    """Legend text: what the backend is, and which model answered."""
    return f"{BACKEND_LABEL.get(name, name)} ({summary['model']})"


def accuracy(summaries: dict[str, dict[str, Any]]) -> Path:
    """Per-question accuracy, plus the all-three-correct rate, as grouped bars."""
    groups = [*QUESTION_IDS, "all three"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.8 / max(1, len(summaries))

    for index, (name, summary) in enumerate(summaries.items()):
        values = [summary["questions"][qid]["accuracy"] for qid in QUESTION_IDS] + [summary["exact_match"]]
        offsets = [i + index * width - 0.4 + width / 2 for i in range(len(groups))]
        bars = ax.bar(offsets, values, width=width, color=BACKEND_COLOUR.get(name, AQUA), label=_label(name, summary))
        ax.bar_label(bars, fmt="%.2f", padding=2, color=INK_2, fontsize=9)

    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy against the gold labels")
    ax.set_title("Same questions, same requests", loc="left")
    ax.legend(loc="lower right")
    return _save(fig, "accuracy.png")


def calibration(summaries: dict[str, dict[str, Any]]) -> Path:
    """Reliability diagram: stated confidence against observed accuracy.

    Points on the diagonal mean the number can be trusted as a probability. Points below it mean the
    backend is overconfident, which is the failure mode that makes automated escalation unsafe.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle=(0, (4, 3)), linewidth=1.2, label="perfectly calibrated")

    for name, summary in summaries.items():
        bins = _pooled_reliability(summary)
        if not bins:
            continue
        ax.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            color=BACKEND_COLOUR.get(name, AQUA),
            label=f"{BACKEND_LABEL.get(name, name)}  ECE {summary['overall_ece']:.3f}",
        )

    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Can you trust the confidence score?", loc="left")
    ax.legend(loc="upper left")
    return _save(fig, "calibration.png")


def deferral(summaries: dict[str, dict[str, Any]]) -> Path:
    """Accuracy on the most confident share of decisions, as that share grows."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for name, summary in summaries.items():
        curve = summary["deferral"]
        ax.plot(
            [100 * point["coverage"] for point in curve],
            [point["accuracy"] for point in curve],
            marker="o",
            markersize=4,
            color=BACKEND_COLOUR.get(name, AQUA),
            label=_label(name, summary),
        )
    ax.set_xlabel("decisions taken automatically (%), most confident first")
    ax.set_ylabel("accuracy of what was taken")
    ax.set_title("Act on the confident routes, review the rest", loc="left")
    ax.legend(loc="lower left")
    return _save(fig, "deferral.png")


def cost_latency(summaries: dict[str, dict[str, Any]]) -> Path:
    """Latency percentiles and cost per 1,000 messages, side by side."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    names = list(summaries)
    width = 0.35

    for index, percentile_name in enumerate(("p50", "p99")):
        offsets = [i + index * width - width / 2 for i in range(len(names))]
        values = [summaries[name]["latency_ms"][percentile_name] for name in names]
        bars = left.bar(
            offsets,
            values,
            width=width,
            color=[BACKEND_COLOUR.get(name, AQUA) for name in names],
            alpha=1.0 if percentile_name == "p50" else 0.45,
            label=percentile_name,
        )
        left.bar_label(bars, fmt="%.0f", padding=2, color=INK_2, fontsize=9)

    left.set_xticks(range(len(names)), [BACKEND_LABEL.get(name, name) for name in names])
    left.set_yscale("log")
    left.set_ylabel("latency per message (ms, log scale)")
    left.set_title("Time to route one request", loc="left")
    left.legend(loc="upper left")

    costs = [summaries[name]["cost_usd_per_1k"] for name in names]
    bars = right.bar(range(len(names)), costs, width=0.5, color=[BACKEND_COLOUR.get(n, AQUA) for n in names])
    right.bar_label(bars, labels=[f"${c:,.2f}" for c in costs], padding=3, color=INK_2, fontsize=10)
    right.set_xticks(range(len(names)), [BACKEND_LABEL.get(name, name) for name in names])
    right.set_ylabel("USD per 1,000 requests")
    right.set_ylim(0, max([*costs, 0.01]) * 1.3)
    right.set_title("Cost of routing one thousand requests", loc="left")
    return _save(fig, "cost_latency.png")


def difficulty(summaries: dict[str, dict[str, Any]]) -> Path:
    """Exact-match rate split by how the message was written."""
    order = ["clear", "terse", "negated", "mixed", "noisy"]
    present = [tag for tag in order if any(tag in s["by_difficulty"] for s in summaries.values())]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.8 / max(1, len(summaries))

    for index, (name, summary) in enumerate(summaries.items()):
        values = [summary["by_difficulty"].get(tag, {}).get("exact_match", 0.0) for tag in present]
        offsets = [i + index * width - 0.4 + width / 2 for i in range(len(present))]
        bars = ax.bar(offsets, values, width=width, color=BACKEND_COLOUR.get(name, AQUA), label=_label(name, summary))
        ax.bar_label(bars, fmt="%.2f", padding=2, color=INK_2, fontsize=9)

    counts = {tag: int(next(iter(summaries.values()))["by_difficulty"].get(tag, {}).get("n", 0)) for tag in present}
    ax.set_xticks(range(len(present)), [f"{tag}\nn={counts[tag]}" for tag in present])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("all three answers correct")
    ax.set_title("Where each router struggles", loc="left")
    ax.legend(loc="lower right")
    return _save(fig, "difficulty.png")


def routing_errors(summaries: dict[str, dict[str, Any]]) -> Path:
    """Where each router's mistakes go: too expensive, or too weak.

    Accuracy alone treats these as the same failure. They are not. Overspending bills you on every request it
    happens to; underspending hands the user a worse answer. A router is chosen by which of the two it
    prefers, so the picture separates them.
    """
    names = list(summaries)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.8 / max(1, len(names))
    kinds = ("correct", "overspend_rate", "underspend_rate")
    titles = ("routed correctly", "too expensive", "too weak")

    for index, name in enumerate(names):
        route = summaries[name]["questions"][ROUTE_ID]
        values = [route["accuracy"], route["overspend_rate"], route["underspend_rate"]]
        offsets = [i + index * width - 0.4 + width / 2 for i in range(len(kinds))]
        bars = ax.bar(
            offsets, values, width=width, color=BACKEND_COLOUR.get(name, AQUA), label=_label(name, summaries[name])
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, color=INK_2, fontsize=9)

    ax.set_xticks(range(len(kinds)), titles)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of the 180 requests")
    ax.set_title("A wrong route costs money or costs quality", loc="left")
    ax.legend(loc="upper right")
    return _save(fig, "routing_errors.png")


def tier_share(summaries: dict[str, dict[str, Any]]) -> Path:
    """What each router actually sends where, against the true mix."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    series = {"gold labels": None} | dict.fromkeys(summaries)
    width = 0.8 / len(series)

    gold_share = _gold_share(summaries)
    for index, name in enumerate(series):
        if name == "gold labels":
            values = [gold_share[tier] for tier in TIER_ORDER]
            colour, label = NEUTRAL, "true mix"
        else:
            share = summaries[name]["questions"][ROUTE_ID]["share"]
            values = [share[tier] for tier in TIER_ORDER]
            colour, label = BACKEND_COLOUR.get(name, AQUA), BACKEND_LABEL.get(name, name)
        offsets = [i + index * width - 0.4 + width / 2 for i in range(len(TIER_ORDER))]
        bars = ax.bar(offsets, values, width=width, color=colour, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=2, color=INK_2, fontsize=9)

    ax.set_xticks(range(len(TIER_ORDER)), list(TIER_ORDER))
    ax.set_ylim(0, max(gold_share.values()) * 2.0)
    ax.set_ylabel("share of requests routed here")
    ax.set_title("Where the traffic actually goes", loc="left")
    ax.legend(loc="upper right", ncol=3)
    return _save(fig, "tier_share.png")


def _gold_share(summaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Recover the true tier mix from any backend's confusion matrix (the row totals are the gold counts)."""
    confusion = next(iter(summaries.values()))["questions"][ROUTE_ID]["confusion"]
    totals = {tier: sum(confusion.get(tier, {}).values()) for tier in TIER_ORDER}
    n = sum(totals.values()) or 1
    return {tier: count / n for tier, count in totals.items()}


def ablation(rows: list[dict[str, Any]], reference: dict[str, Any] | None = None) -> Path:
    """How far the route moves when only the tier descriptions change.

    The model, the requests and the labels are identical across these bars. Everything that differs is three
    sentences of prose, which makes the spread a measure of how much of a router's accuracy is really a
    property of its prompt.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    names = [row["variant"] for row in rows]
    values = [row["macro_f1"] for row in rows]
    best = max(values)
    colours = [BLUE if value == best else "#9dc2ec" for value in values]

    bars = ax.bar(range(len(names)), values, width=0.6, color=colours)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=INK_2, fontsize=10)

    if reference is not None:
        ax.axhline(reference["macro_f1"], color=ORANGE, linewidth=2, linestyle=(0, (4, 3)))
        ax.text(
            len(names) - 0.5,
            reference["macro_f1"] + 0.012,
            f"{reference['label']}  {reference['macro_f1']:.3f}",
            color=ORANGE,
            fontsize=11,
            ha="right",
        )

    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, max(best, reference["macro_f1"] if reference else 0) * 1.35)
    ax.set_ylabel("macro F1 on the route")
    ax.set_title("Same model, same requests, different three sentences", loc="left")
    return _save(fig, "ablation.png")


def _pooled_reliability(summary: dict[str, Any]) -> list[dict[str, float]]:
    """Merge the per-question reliability bins into one curve per backend.

    Per-question diagrams on 150 messages have bins with two or three members, which is noise. The
    per-question error is still reported as a number; the picture pools them.
    """
    merged: dict[tuple[float, float], dict[str, float]] = {}
    for qid in QUESTION_IDS:
        for entry in summary["questions"][qid]["reliability"]:
            key = (entry["lower"], entry["upper"])
            bucket = merged.setdefault(key, {"count": 0.0, "confidence": 0.0, "accuracy": 0.0})
            bucket["count"] += entry["count"]
            bucket["confidence"] += entry["confidence"] * entry["count"]
            bucket["accuracy"] += entry["accuracy"] * entry["count"]
    out = []
    for (lower, upper), bucket in sorted(merged.items()):
        del lower, upper
        if bucket["count"] >= 3:
            out.append(
                {
                    "confidence": bucket["confidence"] / bucket["count"],
                    "accuracy": bucket["accuracy"] / bucket["count"],
                    "count": bucket["count"],
                }
            )
    return out


def render_all(metrics_path: Path) -> list[Path]:
    """Render every figure from the saved results.

    The ablation figure is included when `ablation.json` sits beside the metrics, so re-rendering after a
    re-run cannot leave its reference line pointing at a stale number.
    """
    summaries = json.loads(metrics_path.read_text(encoding="utf-8"))
    ordered = {name: summaries[name] for name in ("laya", "openai") if name in summaries}
    figures = [
        routing_errors(ordered),
        tier_share(ordered),
        accuracy(ordered),
        calibration(ordered),
        deferral(ordered),
        cost_latency(ordered),
        difficulty(ordered),
    ]

    ablation_path = metrics_path.parent / "ablation.json"
    if ablation_path.exists():
        reference = None
        if "openai" in ordered:
            reference = {
                "label": ordered["openai"]["model"],
                "macro_f1": ordered["openai"]["questions"][ROUTE_ID]["macro_f1"],
            }
        figures.append(ablation(json.loads(ablation_path.read_text(encoding="utf-8")), reference))
    return figures
