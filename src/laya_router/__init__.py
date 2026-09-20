"""One routing job, two brains: a resident 421M Laya decision model and an OpenAI LLM classifier."""

from laya_router.questions import BOOLEAN_IDS, QUESTION_IDS, QUESTIONS, TIERS
from laya_router.schema import Decision, RouteResult

__version__ = "0.1.0"
__all__ = [
    "BOOLEAN_IDS",
    "QUESTIONS",
    "QUESTION_IDS",
    "TIERS",
    "Decision",
    "RouteResult",
]
