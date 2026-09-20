"""What every triage backend must provide."""

from typing import Literal, Protocol, runtime_checkable

from laya_router.schema import TriageResult

BackendName = Literal["laya", "openai"]


@runtime_checkable
class Backend(Protocol):
    """A brain that turns one support message into the shared set of typed decisions."""

    name: BackendName
    model: str

    def warmup(self) -> float:
        """Pay any first-call cost before the caller is told the backend is ready.

        Returns the seconds it took. For Laya this is the first MPS kernel setup, which is the
        difference between a 1.8 s first request and a 40 ms one; for a hosted API it is the TLS
        handshake and connection pool.
        """
        ...

    def classify(self, message: str) -> TriageResult:
        """Answer every question about one message."""
        ...
