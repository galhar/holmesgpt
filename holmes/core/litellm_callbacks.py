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

import logging
import os


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
