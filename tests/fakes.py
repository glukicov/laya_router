"""Test doubles shared by the suite.

Nothing here touches the network or loads a real checkpoint: the point of the backend protocol is
that everything above it can be tested without either.
"""

from typing import Any

from laya_router.backends.base import BackendName
from laya_router.questions import BOOLEAN_IDS, QUEUES
from laya_router.schema import Decision, TriageResult


class FakeBackend:
    """A backend that answers from a lookup table, so tests can assert on exact outputs."""

    name: BackendName = "laya"
    model = "fake-checkpoint"
    device = "cpu"
    load_seconds = 1.5

    def __init__(self, answers: dict[str, dict[str, str]] | None = None) -> None:
        self.answers = answers or {}
        self.seen: list[str] = []
        self.warmup_calls = 0
        self.warmup_seconds = 0.0

    def warmup(self) -> float:
        self.warmup_calls += 1
        self.warmup_seconds = 0.25
        return self.warmup_seconds

    def classify(self, message: str) -> TriageResult:
        self.seen.append(message)
        chosen = self.answers.get(message, {"queue": "billing", "urgent": "false", "needs_human": "true"})
        decisions = {
            "queue": Decision(
                answer=chosen["queue"],
                confidence=0.9,
                probabilities={name: (0.9 if name == chosen["queue"] else 0.02) for name in QUEUES},
            )
        }
        for qid in BOOLEAN_IDS:
            decisions[qid] = Decision(answer=chosen[qid], confidence=0.8, probabilities={chosen[qid]: 0.8})
        return TriageResult(
            backend="laya",
            model=self.model,
            decisions=decisions,
            latency_ms=42.0,
            input_tokens=77,
        )


class FakeCompletions:
    """Stand-in for `client.chat.completions`, recording the request and replaying a canned reply."""

    def __init__(self, payload: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> None:
        self.payload = payload
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = type("Message", (), {"content": self.payload})()
        choice = type("Choice", (), {"message": message})()
        usage = type("Usage", (), {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens})()
        return type("Response", (), {"choices": [choice], "usage": usage})()


class FakeOpenAIClient:
    """Minimal `OpenAI()` shape: only `chat.completions.create` is ever used."""

    def __init__(self, payload: str, **kwargs: Any) -> None:
        self.completions = FakeCompletions(payload, **kwargs)
        self.chat = type("Chat", (), {"completions": self.completions})()
