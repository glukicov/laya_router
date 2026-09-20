"""Interchangeable triage backends.

`build` is the only entry point the service and the evaluation use, so adding a third brain means
adding a module here and one line below, not touching either caller.
"""

from laya_router.backends.base import Backend, BackendName

__all__ = ["Backend", "BackendName", "build"]


def build(name: BackendName, **options: object) -> Backend:
    """Construct one backend by name, importing only what that backend needs.

    The imports are deliberately local: `laya` pulls in torch and a 1.7 GB model, and `openai`
    needs a key. A process that only serves one of them should pay for only one of them.
    """
    if name == "laya":
        from laya_router.backends.laya_backend import LayaBackend

        return LayaBackend(**options)  # ty: ignore[invalid-argument-type]
    if name == "openai":
        from laya_router.backends.openai_backend import OpenAIBackend

        return OpenAIBackend(**options)  # ty: ignore[invalid-argument-type]
    raise ValueError(f"unknown backend {name!r}; choose 'laya' or 'openai'")
