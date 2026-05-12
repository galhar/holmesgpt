"""Tool-execution interceptor: wraps each ``Tool`` so its ``_invoke`` delegates
to a ``ToolCallSimulator`` instead of hitting a real backend.

Design
------
HolmesGPT's ``Tool`` (in ``holmes/core/tools.py``) splits execution into two
methods:

  * ``invoke(params, context)``  — public; does parameter coercion, approval
                                   checks, calls ``_invoke``, then runs the
                                   transformer chain (``_apply_transformers``).
  * ``_invoke(params, context)`` — abstract; the actual backend call.

We override **only `_invoke`** on a wrapper class. That way the agent under
test sees the same tool name / description / parameters, and — critically —
the ``llm_summarize`` transformer (and any other configured transformers)
still runs on simulated output. Without this, summary-transformer-equipped
tools would silently behave differently in simulated vs. real eval modes.

Why a class wrapper rather than monkey-patching ``_invoke`` per instance?
Reversibility-by-construction (the original toolset is never mutated), and we
can use a wrapped executor in one test while another test still uses the real
executor in the same pytest session.

Drift-detection notes for porters
---------------------------------
Tool is a Pydantic ``BaseModel`` with several fields. ``SimulatedTool.wrap()``
copies the subset that the LLM and transformer pipeline care about. If
HolmesGPT adds a new required field to ``Tool`` upstream, ``wrap()`` will fail
at construction with a Pydantic ValidationError — add the new field to the
copy list. If ``Tool`` switches from ``_invoke`` being abstract to having a
default implementation, no drift to handle (subclassing still works); if
``Tool.invoke`` itself is overridden by a new subclass *and that override
matters semantically*, our wrapper will lose it — see CAVEATS.md drift #5.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import Field, PrivateAttr

from holmes.core.tools import (
    StructuredToolResult,
    Tool,
    ToolInvokeContext,
    Toolset,
)
from holmes.core.tools_utils.tool_executor import ToolExecutor

from tests.llm.simulation.simulator import ToolCallSimulator

logger = logging.getLogger(__name__)


class SimulatedTool(Tool):
    """Tool subclass whose ``_invoke`` delegates to a ToolCallSimulator.

    The wrapper inherits ``Tool.invoke`` from the base class, so parameter
    coercion, approval checks, the transformer chain, and timing logs all
    behave identically to a real tool — which means the eval is testing the
    same end-to-end pipeline the production agent uses.
    """

    # We intentionally don't declare ``simulator`` as a Pydantic field because
    # ``ToolCallSimulator`` is not a BaseModel (it holds an LLM client). Using
    # PrivateAttr keeps Pydantic out of the way and avoids serialization issues.
    _simulator: ToolCallSimulator = PrivateAttr()

    @classmethod
    def wrap(
        cls,
        original: Tool,
        simulator: ToolCallSimulator,
    ) -> "SimulatedTool":
        """Return a SimulatedTool whose ``_invoke`` defers to ``simulator``.

        Copies the fields the LLM and the transformer chain care about. The
        ``transformers`` field is preserved so e.g. ``llm_summarize`` still
        runs on simulated output.
        """
        # Use model_dump(...) on a per-field basis to avoid copying private
        # state from the original (e.g. _transformer_instances cache, which
        # will be re-initialized by the new instance's model_post_init).
        kwargs: Dict[str, Any] = {
            "name": original.name,
            "description": original.description,
            "parameters": original.parameters,
            "user_description": original.user_description,
            "icon_url": original.icon_url,
            "transformers": original.transformers,
        }
        # ``restricted`` is not always present on every Tool subclass instance,
        # but it is on the base; copy if available.
        if hasattr(original, "restricted"):
            kwargs["restricted"] = original.restricted

        wrapped = cls(**kwargs)
        wrapped._simulator = simulator
        return wrapped

    # ``Tool._invoke`` is abstract; supply our concrete implementation.
    def _invoke(
        self,
        params: Dict,
        context: ToolInvokeContext,
    ) -> StructuredToolResult:
        logger.info(
            "[sim-intercept] agent called %s(%s) → routing to simulator",
            self.name,
            params,
        )
        result = self._simulator.simulate(self.name, params)
        logger.info(
            "[sim-intercept] simulator returned status=%s data_len=%d",
            getattr(result.status, "value", result.status),
            len(result.get_stringified_data()) if result.data is not None else 0,
        )
        return result

    def get_parameterized_one_liner(self, params: Dict) -> str:
        # Keep the simulated tag visible in logs so it's obvious at a glance
        # which tools were faked.
        return f"<simulated> {self.name} {params}"


def build_simulated_tool_executor(
    real_executor: ToolExecutor,
    simulator: ToolCallSimulator,
    only_tools: Optional[set[str]] = None,
) -> ToolExecutor:
    """Return a NEW ``ToolExecutor`` whose enabled toolsets contain
    ``SimulatedTool`` wrappers in place of the original tools.

    The original ``real_executor`` is left untouched, so other tests in the
    same pytest session can still use it for live runs.

    Args:
        real_executor: The executor built by ``TestToolsetManager`` (same as
            in ``test_ask_holmes``).
        simulator: The ``ToolCallSimulator`` to delegate to.
        only_tools: Optional whitelist of tool names to wrap; tools not in
            this set are kept as-is (real). Useful if you ever want hybrid
            mode (e.g. simulate Prometheus but use a real ``echo`` tool).
            Default ``None`` means wrap every tool.
    """
    new_toolsets: list[Toolset] = []
    for ts in real_executor.toolsets:
        wrapped_tools: list[Tool] = []
        for tool in ts.tools:
            if only_tools is not None and tool.name not in only_tools:
                wrapped_tools.append(tool)
                continue
            wrapped_tools.append(SimulatedTool.wrap(tool, simulator))

        # ``model_copy(update=...)`` produces a shallow copy with the listed
        # fields replaced. This is safer than mutating ``ts.tools`` in place
        # because it leaves the original toolset usable for other tests.
        new_ts = ts.model_copy(update={"tools": wrapped_tools})
        # Preserve the toolset's enabled status so ToolExecutor surfaces it.
        new_ts.status = ts.status
        new_ts.enabled = ts.enabled
        new_toolsets.append(new_ts)

    return ToolExecutor(new_toolsets)
