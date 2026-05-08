"""Auto-install Langfuse / Opik LiteLLM callbacks at module-import time
when their respective env vars are set.

Without this, only ``holmes ask --trace langfuse,opik`` produces traces
(via ``TracingFactory.create_tracer`` → ``Tracer.wrap_llm()``). Every
other code path that calls ``litellm.completion()`` directly — pytest
LLM evals, the operator, the FastAPI server — would silently bypass
tracing.

This module registers the global LiteLLM callbacks unconditionally on
import, so every ``litellm.completion()`` call is traced as long as the
env vars exist. Each provider is independent: missing creds for one
doesn't disable the other.

Idempotent — safe to import multiple times. Imports are guarded so
missing SDK packages don't break Holmes startup.

See ``notes/observability-langfuse-opik.md`` for the underlying recipe
and ``notes/holmesgpt-tracing-extension.md`` for how this fits with the
existing ``--trace`` CLI flag (the two paths both end up registering the
same callbacks; the idempotence checks prevent double-emission).
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from typing import Any, Dict, Iterator, Optional


# Per-task session-id propagated to LiteLLM metadata so Langfuse / Opik can
# group all completions from one logical agent run under a single Session
# (Langfuse) or Thread (Opik). Read by build_session_metadata() below.
_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "holmes_session_id", default=None
)
_current_session_user: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "holmes_session_user", default=None
)


@contextlib.contextmanager
def holmes_session(
    session_id: str, user_id: Optional[str] = None
) -> Iterator[str]:
    """Tag every ``litellm.completion()`` call inside this block with
    ``session_id`` so Langfuse Sessions / Opik Threads group them.

    Usage:
        with holmes_session("test:01_how_many_pods"):
            holmes.investigate(...)
    """
    s_token = _current_session_id.set(session_id)
    u_token = _current_session_user.set(user_id) if user_id else None
    try:
        yield session_id
    finally:
        _current_session_id.reset(s_token)
        if u_token is not None:
            _current_session_user.reset(u_token)


def build_session_metadata() -> Dict[str, Any]:
    """If a holmes_session is active, return the metadata kwargs to pass to
    ``litellm.completion(metadata=...)`` so the Langfuse callback groups the
    trace correctly. Returns ``{}`` if no session is active so callers can
    safely splat it into ``**kwargs``.

    Keys produced (only when session_id is set):
      - ``session_id``: provider-agnostic key.
      - ``langfuse_session_id``: Langfuse Sessions UI grouping key
        (https://langfuse.com/docs/tracing-features/sessions).
      - ``langfuse_user_id``: optional user attribution.
    """
    sid = _current_session_id.get()
    if not sid:
        return {}
    md: Dict[str, Any] = {
        "session_id": sid,
        "langfuse_session_id": sid,
    }
    uid = _current_session_user.get()
    if uid:
        md["langfuse_user_id"] = uid
    return md


def build_opik_args() -> Dict[str, Any]:
    """If a holmes_session is active, return the ``opik_args`` kwarg to pass
    to ``litellm.completion(opik_args=...)`` so the Opik decorator tags the
    trace with ``thread_id`` (= Opik's "Thread" grouping key).

    Returns ``{}`` if no session is active. Opik expects this as a separate
    top-level kwarg, not nested in ``metadata`` — see
    ``opik/decorator/opik_args/api_classes.py::OpikArgs.from_dict``. The
    public dict keys are ``"trace"`` and ``"span"`` (not ``"trace_args"``;
    that's the Pydantic field name, but the dict accepts the public form).
    """
    sid = _current_session_id.get()
    if not sid:
        return {}
    return {"opik_args": {"trace": {"thread_id": sid}}}


def _install_langfuse_callback() -> bool:
    """Append ``"langfuse"`` to ``litellm.success_callback`` and
    ``litellm.failure_callback`` if Langfuse env vars are set.
    Returns True if installed, False otherwise.
    """
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        return False
    try:
        import langfuse  # noqa: F401  (verifies SDK is importable)
    except ImportError:
        return False

    import litellm

    cur_success = list(litellm.success_callback or [])
    if "langfuse" not in cur_success:
        cur_success.append("langfuse")
        litellm.success_callback = cur_success

    cur_failure = list(litellm.failure_callback or [])
    if "langfuse" not in cur_failure:
        cur_failure.append("langfuse")
        litellm.failure_callback = cur_failure

    return True


def _install_opik_callback() -> bool:
    """Monkey-patch ``litellm.completion`` with Opik's
    ``track_completion`` decorator if Opik env vars are set.
    Returns True if installed, False otherwise.

    Requires both ``OPIK_API_KEY`` and ``OPIK_WORKSPACE`` (the latter
    is critical: ``opik.configure()`` hangs on stdin asking for it
    when missing, which would deadlock non-interactive contexts).
    """
    if not (
        os.environ.get("OPIK_API_KEY") and os.environ.get("OPIK_WORKSPACE")
    ):
        return False
    try:
        from opik.integrations.litellm import track_completion
    except ImportError:
        return False

    import litellm

    if getattr(litellm.completion, "_opik_tracked", False):
        return True  # already wrapped (idempotent)

    project = os.environ.get("OPIK_PROJECT_NAME", "HolmesGPT")
    wrapped = track_completion(project_name=project)(litellm.completion)
    try:
        wrapped._opik_tracked = True  # type: ignore[attr-defined]
    except AttributeError:
        pass
    litellm.completion = wrapped

    return True


def install_global_litellm_callbacks() -> None:
    """Install both callbacks if their env vars are present. Logs at
    info level so an operator can confirm tracing is wired up; downgrades
    to debug if the SDK isn't installed (which is fine — the user just
    chose not to install that provider).
    """
    installed = []
    if _install_langfuse_callback():
        installed.append("langfuse")
    if _install_opik_callback():
        installed.append("opik")
    if installed:
        logging.info(
            "LiteLLM auto-trace enabled for: %s",
            ", ".join(installed),
        )


# Side-effect on import: install whatever the env permits. Cheap when
# nothing is configured — just two os.environ.get() calls.
install_global_litellm_callbacks()
