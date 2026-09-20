"""The resident Laya backend: one 421M encoder, one forward pass per message.

Keeping the model resident is the whole trick. Constructing it and moving it onto Apple MPS costs
tens of seconds; once warm, every question in the schema is answered in a single batched forward
pass with no tokens generated, so there is nothing to parse and nothing to hallucinate.
"""

import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

# Laya is torch-only. Disable Transformers' TensorFlow probe before importing the SDK.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from huggingface_hub import snapshot_download

import laya
from laya_router.backends.base import BackendName
from laya_router.questions import QUESTIONS
from laya_router.schema import Decision, TriageResult

MODEL_ID = "convaiinnovations/laya"

# The hub repo bundles three checkpoints (~2.37 GB). This filter fetches only the English root one,
# which is the 843 MB we actually run.
MODEL_FILES = ("model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*")

WARMUP_MESSAGE = "Write a Python function that parses this CSV and returns the rows above a threshold."


def load_agent(device: str | None = None) -> Any:
    """Download the English checkpoint if it is not cached, then build one agent."""
    model_path = Path(snapshot_download(repo_id=MODEL_ID, allow_patterns=list(MODEL_FILES)))
    return laya.load(model_path, device=device)  # ty: ignore[invalid-argument-type]


def _decision(qid: str, answer: dict[str, Any]) -> Decision:
    """Map one Laya answer onto the shared contract.

    Laya returns a different shape per question type. `choice` carries a named option and a
    probability per option; `noul` carries `P(true)` and reports confidence as the distance from the
    coin flip, which is what makes a yes/no answer comparable with a six-way one.
    """
    if answer["type"] == "choice":
        probabilities = {k: float(v) for k, v in answer["probabilities"].items()}
        # Deliberately NOT the SDK's `confidence` field. For a choice question that field is normalised
        # entropy, 1 - H(p)/log(k): a measure of how peaked the distribution is, which is not on the same
        # scale as "the probability this answer is right". The evaluation compares Laya's confidence against
        # a generative model's self-reported P(correct), so both sides must mean the same thing, and the
        # decision head already provides it as the probability of the chosen option.
        return Decision(
            answer=str(answer["choice"]),
            confidence=max(probabilities.values()),
            probabilities=probabilities,
        )
    if answer["type"] == "noul":
        p_true = float(answer["noul"])
        return Decision(
            answer="true" if p_true >= 0.5 else "false",
            confidence=max(p_true, 1.0 - p_true),
            probabilities={"true": p_true, "false": 1.0 - p_true},
        )
    raise ValueError(f"question {qid!r} returned unsupported type {answer['type']!r}")


class LayaBackend:
    """Own one resident model and serialize access to the shared accelerator."""

    name: BackendName = "laya"

    def __init__(self, device: str | None = None, agent: Any | None = None) -> None:
        started = perf_counter()
        self.agent = agent if agent is not None else load_agent(device)
        self.load_seconds = perf_counter() - started
        self.device = str(getattr(self.agent, "device", device or "cpu"))
        self.model = MODEL_ID
        self.warmup_seconds = 0.0
        # One model on one device: concurrent forward passes contend rather than parallelise, and on
        # MPS they are a source of flaky failures. Serialising is both faster and safer here.
        self._lock = Lock()

    def warmup(self) -> float:
        """Run one representative message so the first real request is not the slow one."""
        started = perf_counter()
        self.classify(WARMUP_MESSAGE)
        self.warmup_seconds = perf_counter() - started
        return self.warmup_seconds

    def classify(self, message: str) -> TriageResult:
        """Answer every question about `message` in a single forward pass."""
        with self._lock, torch.inference_mode():
            started = perf_counter()
            raw = self.agent.predict({"message": message}, QUESTIONS)
            latency_ms = (perf_counter() - started) * 1_000

        return TriageResult(
            backend="laya",
            model=self.model,
            decisions={qid: _decision(qid, answer) for qid, answer in raw["answers"].items()},
            latency_ms=latency_ms,
            input_tokens=int(raw.get("usage", {}).get("input_tokens", 0)),
            output_tokens=0,
            # Local inference has no per-request invoice. The hardware it runs on is not free, and
            # docs/EVAL.md puts a number on that rather than pretending otherwise.
            cost_usd=0.0,
        )
