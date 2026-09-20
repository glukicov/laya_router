"""The one test that runs the real checkpoint.

Marked slow and excluded from CI: it downloads 843 MB and needs an accelerator to be quick. It is kept
because every other test uses a fake, and a fake cannot catch the SDK changing the shape of its answers.

    uv run pytest -m slow
"""

import pytest

from laya_router.questions import QUEUES
from laya_router.schema import Decision


@pytest.mark.slow
def test_real_checkpoint_answers_every_question_in_one_pass() -> None:
    from laya_router.backends.laya_backend import LayaBackend

    backend = LayaBackend()
    result = backend.classify("I was billed twice for invoice 4411. Please refund the duplicate charge today.")

    assert set(result.decisions) == {"queue", "urgent", "needs_human"}
    assert all(isinstance(d, Decision) for d in result.decisions.values())
    assert result.decisions["queue"].answer in QUEUES
    # The model reads the message but writes nothing, so a token bill would be input-only.
    assert result.input_tokens > 0
    assert result.output_tokens == 0

    # Probabilities must be a distribution, or the calibration analysis is meaningless.
    total = sum(result.decisions["queue"].probabilities.values())
    assert total == pytest.approx(1.0, abs=0.01)

    # Warm inference must be far quicker than the cold construction it follows.
    backend.warmup()
    warm = backend.classify("The API is returning 500 errors after today's deploy.")
    assert warm.latency_ms < 2_000
