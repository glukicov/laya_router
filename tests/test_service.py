"""The service contract: same request and same response, whichever brain is behind it."""

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from laya_router.service import create_app
from tests.fakes import FakeBackend


def test_backend_is_built_and_warmed_once_before_ready() -> None:
    backend = FakeBackend()
    builds: list[int] = []

    def builder() -> FakeBackend:
        builds.append(1)
        return backend

    with TestClient(create_app(backend_builder=builder)) as client:
        health = client.get("/health")
        first = client.post("/route", json={"message": "Design our sharding strategy."})

    assert len(builds) == 1
    assert backend.warmup_calls == 1
    assert health.json()["status"] == "ready"
    assert health.json()["warmup_seconds"] == 0.25
    assert first.status_code == 200
    assert first.json()["decisions"]["tier"]["answer"] == "medium"


def test_route_returns_every_question_in_the_schema() -> None:
    with TestClient(create_app(backend_builder=FakeBackend)) as client:
        body = client.post("/route", json={"message": "hello"}).json()
        schema = client.get("/questions").json()

    assert set(body["decisions"]) == set(schema)
    assert set(schema) == {"tier", "needs_tools", "is_sensitive"}


def test_invalid_requests_are_rejected_before_the_model_runs() -> None:
    backend = FakeBackend()

    with TestClient(create_app(backend_builder=lambda: backend)) as client:
        empty = client.post("/route", json={"message": ""})
        extra = client.post("/route", json={"message": "valid", "unexpected": True})

    assert empty.status_code == 422
    assert extra.status_code == 422
    # Neither malformed request ever reached the model.
    assert backend.seen == []


def test_concurrent_requests_all_get_answered() -> None:
    backend = FakeBackend()

    with (
        TestClient(create_app(backend_builder=lambda: backend)) as client,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        responses = list(pool.map(lambda m: client.post("/route", json={"message": m}), ["a", "b", "c", "d"]))

    assert [r.status_code for r in responses] == [200] * 4
    assert set(backend.seen) >= {"a", "b", "c", "d"}
