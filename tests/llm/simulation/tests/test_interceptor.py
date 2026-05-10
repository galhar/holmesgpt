"""Tests for SimulatedTool wrapping and the build_simulated_tool_executor flow.

Includes the critical drift-detection test that the llm_summarize transformer
is preserved on simulated output (see CAVEATS.md drift risk #2).
"""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock

import pytest

from holmes.core.llm import LLM
from holmes.core.tools import (
    StructuredToolResult,
    StructuredToolResultStatus,
    Tool,
    ToolInvokeContext,
    ToolParameter,
    Toolset,
    ToolsetStatusEnum,
)
from holmes.core.tools_utils.tool_executor import ToolExecutor
from holmes.core.transformers import Transformer
from holmes.core.transformers.base import BaseTransformer

from tests.llm.simulation.interceptor import (
    SimulatedTool,
    build_simulated_tool_executor,
)
from tests.llm.simulation.recording import Recording


class _RealTool(Tool):
    """A trivial concrete Tool used as the wrapping target."""

    def _invoke(
        self, params: Dict, context: ToolInvokeContext
    ) -> StructuredToolResult:
        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=f"REAL: {params}",
        )

    def get_parameterized_one_liner(self, params: Dict) -> str:
        return f"real {self.name}({params})"


def _ctx(tool_name: str = "x") -> ToolInvokeContext:
    # spec=LLM makes the MagicMock pass Pydantic's isinstance validation,
    # which is gated by ToolInvokeContext.model_config.arbitrary_types_allowed.
    return ToolInvokeContext(
        llm=MagicMock(spec=LLM),
        max_token_count=4096,
        tool_name=tool_name,
        tool_call_id="tc_test",
        user_approved=True,
    )


def _empty_recording() -> Recording:
    return Recording(
        system_prompt="s",
        user_prompt="u",
        tool_calls=[],
        ground_truth_diagnosis="g",
    )


def _stub_simulator(canned: StructuredToolResult) -> MagicMock:
    sim = MagicMock()
    sim.simulate.return_value = canned
    return sim


# ----- SimulatedTool.wrap ---------------------------------------------------


def test_wrap_copies_identity_fields() -> None:
    real = _RealTool(
        name="kubectl_get_pods",
        description="get pods",
        parameters={"namespace": ToolParameter(type="string", required=True)},
        user_description="kubectl get pods -n {{ namespace }}",
    )
    sim = _stub_simulator(
        StructuredToolResult(status=StructuredToolResultStatus.SUCCESS, data="ok")
    )
    wrapped = SimulatedTool.wrap(real, sim)
    assert wrapped.name == real.name
    assert wrapped.description == real.description
    assert wrapped.parameters == real.parameters
    assert wrapped.user_description == real.user_description


def test_wrapped_invoke_calls_simulator() -> None:
    real = _RealTool(name="t", description="d")
    canned = StructuredToolResult(
        status=StructuredToolResultStatus.SUCCESS, data="SIMULATED"
    )
    sim = _stub_simulator(canned)
    wrapped = SimulatedTool.wrap(real, sim)
    out = wrapped.invoke(params={"a": 1}, context=_ctx())
    assert out.status == StructuredToolResultStatus.SUCCESS
    assert out.data == "SIMULATED"
    sim.simulate.assert_called_once_with("t", {"a": 1})


# ----- transformer-preservation: the key drift-detection test --------------


class _UpperCaseTransformer(BaseTransformer):
    """A trivial transformer that uppercases the tool's stringified data."""

    def transform(self, input_text: str) -> str:
        return input_text.upper()

    def should_apply(self, input_text: str) -> bool:
        return len(input_text) > 0

    @property
    def name(self) -> str:
        return "uppercase"


def test_transformer_pipeline_runs_on_simulated_output() -> None:
    """If the original tool declares transformers, those transformers MUST
    still run on simulated output. This is the contract that lets the
    llm_summarize transformer keep working in simulated tests.

    Drift-detection: if this test starts failing after an upstream change,
    SimulatedTool is intercepting at the wrong level (probably overriding
    .invoke() instead of ._invoke()) and the llm_summarize transformer
    would silently be skipped on simulated output too. See
    tests/llm/simulation/CAVEATS.md drift risk #2.
    """
    from holmes.core.transformers import registry as transformer_registry

    transformer_registry.register(_UpperCaseTransformer)
    try:
        real = _RealTool(
            name="t",
            description="d",
            transformers=[Transformer(name="uppercase", config={})],
        )
        canned = StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS, data="hello world"
        )
        sim = _stub_simulator(canned)
        wrapped = SimulatedTool.wrap(real, sim)
        out = wrapped.invoke(params={}, context=_ctx())
        assert out.status == StructuredToolResultStatus.SUCCESS
        assert out.data == "HELLO WORLD", (
            "transformer chain did not run on simulated output — "
            "this means SimulatedTool is intercepting at the wrong level "
            "and the llm_summarize transformer would also be skipped"
        )
    finally:
        if transformer_registry.is_registered("uppercase"):
            transformer_registry.unregister("uppercase")


# ----- build_simulated_tool_executor ---------------------------------------


def _toolset_with_tools(name: str, tools: list[Tool]) -> Toolset:
    ts = Toolset(name=name, description=f"{name} toolset", tools=tools, enabled=True)
    ts.status = ToolsetStatusEnum.ENABLED
    return ts


def test_build_simulated_executor_wraps_all_tools() -> None:
    real_a = _RealTool(name="a", description="d")
    real_b = _RealTool(name="b", description="d")
    ts = _toolset_with_tools("ts1", [real_a, real_b])
    real_executor = ToolExecutor([ts])

    sim = _stub_simulator(
        StructuredToolResult(status=StructuredToolResultStatus.SUCCESS, data="X")
    )
    new_exec = build_simulated_tool_executor(real_executor, sim)

    # All tools in the new executor are SimulatedTool instances.
    for tool in new_exec.tools_by_name.values():
        assert isinstance(tool, SimulatedTool)

    # The real executor is untouched (this is the reversibility property).
    for tool in real_executor.tools_by_name.values():
        assert not isinstance(tool, SimulatedTool)


def test_build_simulated_executor_respects_only_tools_whitelist() -> None:
    real_a = _RealTool(name="a", description="d")
    real_b = _RealTool(name="b", description="d")
    ts = _toolset_with_tools("ts1", [real_a, real_b])
    real_executor = ToolExecutor([ts])

    sim = _stub_simulator(
        StructuredToolResult(status=StructuredToolResultStatus.SUCCESS, data="X")
    )
    new_exec = build_simulated_tool_executor(real_executor, sim, only_tools={"a"})

    assert isinstance(new_exec.tools_by_name["a"], SimulatedTool)
    assert not isinstance(new_exec.tools_by_name["b"], SimulatedTool)
