"""Command line for the service, the evaluation and the figures."""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from laya_router.backends.base import BackendName
from laya_router.data import DATASET, load_requests, read_jsonl
from laya_router.evaluate import RESULTS, adjudicate, run
from laya_router.metrics import score

app = typer.Typer(add_completion=False, help="One triage job, two brains: Laya against an OpenAI classifier.")
eval_app = typer.Typer(add_completion=False, help="Run and score the head-to-head evaluation.")
app.add_typer(eval_app, name="eval")

BackendOption = Annotated[str, typer.Option("--backend", "-b", help="Which brain to use: laya or openai.")]
DatasetOption = Annotated[Path, typer.Option("--dataset", help="Labelled JSONL to evaluate against.")]
ResultsOption = Annotated[Path, typer.Option("--results", help="Directory holding one JSONL per backend.")]


def _load_env() -> None:
    """Read `.env` if it is there, so an API key never has to be pasted into a shell.

    Missing python-dotenv is not an error: the key may already be exported, and the core service
    does not need the evaluation extras installed at all.
    """
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(Path.cwd() / ".env")


def _backend_name(value: str) -> BackendName:
    """Validate the backend name at the boundary rather than deep inside a run."""
    if value not in ("laya", "openai"):
        raise typer.BadParameter("backend must be 'laya' or 'openai'")
    return value


def _build(name: BackendName, device: str | None, model: str | None) -> Any:
    """Construct one backend with only the options that backend understands."""
    from laya_router.backends import build

    _load_env()
    if name == "laya":
        return build("laya", device=None if device in (None, "auto") else device)
    return build("openai", **({"model": model} if model else {}))


@app.command()
def serve(
    backend: BackendOption = "laya",
    host: str = typer.Option("127.0.0.1", help="Interface to bind. The loopback default is local-only."),
    port: int = typer.Option(8000, min=1, max=65_535),
    device: str = typer.Option("auto", help="Torch device for the Laya backend: auto, mps, cuda or cpu."),
    model: str = typer.Option("", help="Model id for the OpenAI backend."),
) -> None:
    """Serve /triage with one warm, resident backend."""
    import uvicorn

    from laya_router.service import create_app

    name = _backend_name(backend)
    _load_env()
    options: dict[str, Any] = {"device": None if device == "auto" else device} if name == "laya" else {}
    if name == "openai" and model:
        options["model"] = model
    # One worker on purpose: a second would duplicate the resident model and contend for the same
    # accelerator, which costs memory and gains nothing.
    uvicorn.run(create_app(backend=name, **options), host=host, port=port, workers=1)


@app.command()
def classify(
    message: Annotated[str, typer.Argument(help="The support message to triage.")],
    backend: BackendOption = "laya",
    device: str = typer.Option("auto"),
    model: str = typer.Option(""),
) -> None:
    """Classify one message without starting a server. Cold on every invocation."""
    instance = _build(_backend_name(backend), device, model or None)
    typer.echo(instance.classify(message).model_dump_json(indent=2))


@eval_app.command("run")
def eval_run(
    backend: BackendOption = "laya",
    dataset: DatasetOption = DATASET,
    limit: int = typer.Option(0, help="Evaluate only the first N requests. 0 means all of them."),
    device: str = typer.Option("auto"),
    model: str = typer.Option(""),
    out: Annotated[Path | None, typer.Option("--out", help="Where to write the per-message rows.")] = None,
) -> None:
    """Classify every labelled request once and save the raw results."""
    name = _backend_name(backend)
    requests = load_requests(dataset, limit=limit or None)
    instance = _build(name, device, model or None)
    warmup_seconds = instance.warmup()
    typer.echo(f"{name} ({instance.model}) warm in {warmup_seconds:.2f}s; {len(requests)} requests", err=True)

    def progress(index: int, _request: Any, _row: dict[str, Any]) -> None:
        if (index + 1) % 25 == 0 or index + 1 == len(requests):
            typer.echo(f"  {index + 1}/{len(requests)}", err=True)

    destination = out if out is not None else RESULTS / f"{name}.jsonl"
    run(instance, requests, out_path=destination, on_result=progress)
    typer.echo(str(destination))


@eval_app.command("report")
def eval_report(
    dataset: DatasetOption = DATASET,
    results: ResultsOption = RESULTS,
) -> None:
    """Score every saved run and write results/metrics.json plus a summary table."""
    requests = load_requests(dataset)
    summaries = {}
    for path in sorted(results.glob("*.jsonl")):
        if path.stem in ("laya", "openai"):
            summaries[path.stem] = score(list(read_jsonl(path)), requests)
    if not summaries:
        raise typer.BadParameter(f"no laya.jsonl or openai.jsonl found in {results}")

    destination = results / "metrics.json"
    destination.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    typer.echo(_summary_table(summaries))
    typer.echo(str(destination), err=True)


@eval_app.command("adjudicate")
def eval_adjudicate(
    dataset: DatasetOption = DATASET,
    model: str = typer.Option("gpt-5", help="A strong model, independent of the one under test."),
    out: Annotated[Path, typer.Option("--out")] = RESULTS / "adjudication.jsonl",
) -> None:
    """Audit the hand-written gold labels with a stronger, independent model."""
    _load_env()
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise typer.BadParameter("OPENAI_API_KEY is not set")
    requests = load_requests(dataset)
    rows = adjudicate(OpenAI(api_key=key, timeout=120.0, max_retries=3), requests, model=model, out_path=out)
    disagreements = [row for row in rows if row["verdict"] == "disagree"]
    typer.echo(f"{len(rows) - len(disagreements)}/{len(rows)} labels upheld by {model}")
    for row in disagreements:
        typer.echo(f"  {row['id']}: {row['field']} -> {row['suggested']} ({row['reason']})")
    typer.echo(str(out), err=True)


@eval_app.command("ablate")
def eval_ablate(
    dataset: DatasetOption = DATASET,
    device: str = typer.Option("auto"),
    out: Annotated[Path, typer.Option("--out")] = RESULTS / "ablation.json",
) -> None:
    """Re-answer the queue question under each wording of the option descriptions."""
    from laya_router.ablation import run as run_ablation

    requests = load_requests(dataset)
    backend = _build("laya", device, None)
    backend.warmup()
    rows = run_ablation(backend, requests)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    header = rows[0]
    typer.echo(
        f"| queue wording | options | macro F1 (own set) | macro F1 (same {header['n_common']} messages) "
        "| answered `other` |"
    )
    typer.echo("|---|---:|---:|---:|---:|")
    for row in rows:
        typer.echo(
            f"| {row['variant']} | {row['options']} | {row['macro_f1']:.3f} (n={row['n']}) "
            f"| {row['macro_f1_common']:.3f} | {row['other_rate']:.1%} |"
        )
    typer.echo(str(out), err=True)


@app.command()
def figures(results: ResultsOption = RESULTS) -> None:
    """Render every figure in docs/figures from results/metrics.json."""
    from laya_router.plots import render_all

    for path in render_all(results / "metrics.json"):
        typer.echo(str(path))


def _summary_table(summaries: dict[str, dict[str, Any]]) -> str:
    """A small markdown table, so a run can be pasted straight into the write-up."""
    header = "| backend | model | exact match | queue F1 | urgent acc | needs_human acc | ECE | p50 ms | $/1k |"
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        q = s["questions"]
        lines.append(
            f"| {name} | {s['model']} | {s['exact_match']:.3f} | {q['queue']['macro_f1']:.3f} "
            f"| {q['urgent']['accuracy']:.3f} | {q['needs_human']['accuracy']:.3f} "
            f"| {s['overall_ece']:.3f} | {s['latency_ms']['p50']:.0f} | ${s['cost_usd_per_1k']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Start the command line."""
    sys.exit(app())
