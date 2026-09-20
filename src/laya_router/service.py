"""The triage service: one endpoint, one backend chosen at startup.

Whichever brain is configured, callers see the same request and the same response, which is the
claim this repo is testing. The backend is built and warmed during the lifespan so the process is
only reported ready once the first real request will be a warm one.
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from laya_router import __version__
from laya_router.backends import build
from laya_router.backends.base import Backend, BackendName
from laya_router.questions import QUESTIONS
from laya_router.schema import TriageResult

LOGGER = logging.getLogger(__name__)
BackendBuilder = Callable[[], Backend]


class TriageRequest(BaseModel):
    """One inbound support message."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=100_000)


class HealthResponse(BaseModel):
    """Readiness plus the start-up costs worth knowing about."""

    status: Literal["ready"]
    backend: str
    model: str
    device: str | None = None
    load_seconds: float = Field(default=0.0, ge=0)
    warmup_seconds: float = Field(default=0.0, ge=0)


def get_backend(request: Request) -> Backend:
    """Return the lifespan-managed backend, or 503 before it is ready."""
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Backend is not ready")
    return backend


router = APIRouter(tags=["triage"])


@router.get("/health", response_model=HealthResponse)
async def health(backend: Annotated[Backend, Depends(get_backend)]) -> HealthResponse:
    """Report readiness once the backend is loaded and warm."""
    return HealthResponse(
        status="ready",
        backend=backend.name,
        model=backend.model,
        device=getattr(backend, "device", None),
        load_seconds=getattr(backend, "load_seconds", 0.0),
        warmup_seconds=getattr(backend, "warmup_seconds", 0.0),
    )


@router.get("/questions")
async def questions() -> dict[str, Any]:
    """Return the schema both backends answer, so a caller can see the contract."""
    return QUESTIONS


@router.post("/triage", response_model=TriageResult)
async def triage(
    request: TriageRequest,
    backend: Annotated[Backend, Depends(get_backend)],
) -> TriageResult:
    """Classify one message with the configured backend."""
    return await run_in_threadpool(backend.classify, request.message)


def create_app(
    backend: BackendName = "laya",
    backend_builder: BackendBuilder | None = None,
    **options: Any,
) -> FastAPI:
    """Create a single-process app that builds and warms exactly one backend at startup.

    `backend_builder` exists so the tests can inject a fake and exercise the whole app without
    downloading 843 MB of weights or spending anything on an API.
    """
    builder: BackendBuilder = backend_builder or (lambda: build(backend, **options))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        instance = await run_in_threadpool(builder)
        warmup_seconds = await run_in_threadpool(instance.warmup)
        app.state.backend = instance
        LOGGER.info("%s ready (%s), warm-up %.3fs", instance.name, instance.model, warmup_seconds)
        yield

    app = FastAPI(title="laya-router", version=__version__, lifespan=lifespan)
    app.include_router(router)
    return app
