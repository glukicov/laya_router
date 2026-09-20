<div align="center">

<h2>A 421M model answers your triage questions in one forward pass. Do you still need an LLM for it?</h2>

[![CI](https://github.com/glukicov/laya_router/actions/workflows/ci.yml/badge.svg)](https://github.com/glukicov/laya_router/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](.python-version)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
<br>
[![Laya](https://img.shields.io/badge/%F0%9F%A4%97%20Model-convaiinnovations%2Flaya-blue)](https://huggingface.co/convaiinnovations/laya)
[![OpenAI](https://img.shields.io/badge/OpenAI-structured%20outputs-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/structured-outputs)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![kind](https://img.shields.io/badge/Kubernetes-kind-326CE5?logo=kubernetes&logoColor=white)](https://kind.sigs.k8s.io)

**[Results](#results) · [Quickstart](#quickstart) · [The experiment](docs/EVAL.md) · [The data](data/README.md)**

</div>

**laya_router — one triage job, two brains.** A support-triage service that turns an inbound message into three
typed decisions (which queue, is it urgent, does a human need to handle it), answered either by
[Laya](https://huggingface.co/convaiinnovations/laya) — a 421M non-autoregressive decision model running on your
laptop — or by a hosted OpenAI classifier with structured outputs. Same endpoint, same response, swap the brain
with a flag. Then a head-to-head on 150 hand-labelled messages: accuracy, calibration, latency and cost.

Laya is not generative. It answers every question in **one forward pass** as probabilities over the options, so
there is no text to parse, nothing to hallucinate, no output tokens to pay for — and a confidence number that
turns out to mean something.

<!-- RESULTS -->

## Quickstart

```bash
git clone https://github.com/glukicov/laya_router && cd laya_router
uv sync --all-extras

# Serve it. First run downloads 843 MB of weights; the model then stays resident.
uv run laya-router serve --backend laya

curl -s localhost:8000/triage -H 'content-type: application/json' \
  -d '{"message":"I was charged twice for invoice 4411. Please refund the duplicate charge."}' | python3 -m json.tool
```

Swap the brain without touching the caller:

```bash
cp .env.example .env   # add your OPENAI_API_KEY
uv run laya-router serve --backend openai
```

Run the whole study yourself:

```bash
uv run laya-router eval run --backend laya --device mps
uv run laya-router eval run --backend openai
uv run laya-router eval report        # the comparison table
uv run laya-router figures            # docs/figures/*.png
```

On Kubernetes (local, CPU-only — a Linux container on macOS cannot reach Apple's MPS backend):

```bash
k8s/kind/up.sh      # builds the image, creates the cluster, serves on localhost:8080
k8s/kind/down.sh
```

## Layout

```
src/laya_router/
  questions.py     the one schema both brains answer; the OpenAI prompt and JSON schema are generated from it
  backends/        laya_backend.py (resident, one forward pass) · openai_backend.py (structured outputs)
  service.py       FastAPI: /triage, /health, /questions
  evaluate.py      run a backend over the labelled set · audit the gold labels with an independent model
  metrics.py       accuracy, macro F1, ECE, deferral curves — dependency-free, so CI scores without extras
  ablation.py      does rewording the options change the answers?
data/              150 labelled messages and how they were written
docs/EVAL.md       the experiment: every number and what it does not show
k8s/kind/          one replica, one resident model, weights mounted from the host cache
```

## Caveats

150 hand-written messages on one laptop, zero-shot on both sides, with no domain calibration — Laya's model card
asks for temperature fitting before operational use, and this evaluation deliberately skips it to measure what
comes out of the box. Differences under ~8 points are inside the noise. The full list is in
[docs/EVAL.md](docs/EVAL.md#limitations).
