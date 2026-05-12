"""Auto-install Langfuse + Opik LiteLLM callbacks on import (when their env
vars are set), and expose ``holmes_session()`` so callers can group all
``litellm.completion()`` calls of one investigation under a single Langfuse
Session / Opik Thread.

Auto-install means the existing ``--trace`` CLI flag is no longer required:
pytest LLM evals, the FastAPI server, the operator — every code path that
calls ``litellm.completion()`` is traced as long as the env vars exist.
Idempotent and SDK-import-guarded so a missing provider never breaks
startup. See ``notes/observability-langfuse-opik.md`` for the underlying
recipe.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from typing import Any, Dict, Iterator, Optional


# Set by holmes_session(); read by build_session_metadata / build_opik_args
# when DefaultLLM.completion() builds the litellm call kwargs.
_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "holmes_session_id", default=None
)


@contextlib.contextmanager
def holmes_session(session_id: str) -> Iterator[str]:
    """Tag every ``litellm.completion()`` inside this block with
    ``session_id`` so Langfuse Sessions / Opik Threads group them.
    """
    token = _current_session_id.set(session_id)
    try:
        yield session_id
    finally:
        _current_session_id.reset(token)


def build_session_metadata() -> Dict[str, Any]:
    """Returns ``metadata`` kwargs for ``litellm.completion()`` so the
    Langfuse callback groups the trace by session, or ``{}`` if no session
    is active. Caller should ``**`` it into kwargs.
    """
    sid = _current_session_id.get()
    if not sid:
        return {}
    return {"session_id": sid, "langfuse_session_id": sid}


def build_opik_args() -> Dict[str, Any]:
    """Returns the ``opik_args`` kwarg so the Opik decorator tags the trace
    with ``thread_id`` (= Opik Threads grouping key), or ``{}`` if no
    session is active.

    Opik wants this as a separate top-level kwarg, not nested in
    ``metadata`` — and the public dict key is ``"trace"`` (not
    ``"trace_args"``); see opik's ``OpikArgs.from_dict``.
    """
    sid = _current_session_id.get()
    if not sid:
        return {}
    return {"opik_args": {"trace": {"thread_id": sid}}}


def _install_langfuse_callback() -> bool:
    """Register ``"langfuse"`` in litellm's success+failure callbacks if
    LANGFUSE_PUBLIC_KEY+SECRET_KEY are set and the SDK is importable."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return False
    try:
        import langfuse  # noqa: F401
    except ImportError:
        return False

    import litellm

    for attr in ("success_callback", "failure_callback"):
        cur = list(getattr(litellm, attr) or [])
        if "langfuse" not in cur:
            setattr(litellm, attr, cur + ["langfuse"])
    return True


def _install_opik_callback() -> bool:
    """Wrap ``litellm.completion`` with Opik's ``track_completion`` decorator
    if OPIK_API_KEY+OPIK_WORKSPACE are set and the SDK is importable.

    OPIK_WORKSPACE is required: without it, ``opik.configure()`` would
    hang on stdin and deadlock non-interactive contexts.
    """
    if not (os.environ.get("OPIK_API_KEY") and os.environ.get("OPIK_WORKSPACE")):
        return False
    try:
        from opik.integrations.litellm import track_completion
    except ImportError:
        return False

    import litellm

    if getattr(litellm.completion, "_opik_tracked", False):
        return True

    wrapped = track_completion(project_name=os.environ.get("OPIK_PROJECT_NAME", "HolmesGPT"))(
        litellm.completion
    )
    try:
        wrapped._opik_tracked = True  # type: ignore[attr-defined]
    except AttributeError:
        pass
    litellm.completion = wrapped
    return True


def install_global_litellm_callbacks() -> None:
    installed = [
        name for name, ok in (
            ("langfuse", _install_langfuse_callback()),
            ("opik", _install_opik_callback()),
        )
        if ok
    ]
    if installed:
        logging.info("LiteLLM auto-trace enabled for: %s", ", ".join(installed))


# Side-effect on import. Cheap when nothing is configured.
install_global_litellm_callbacks()
