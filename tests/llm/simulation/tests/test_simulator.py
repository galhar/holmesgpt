"""Tests for the LLM-as-simulator inner loop, JSON parsing, and meta-tools."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus
from tests.llm.simulation.recording import (
    RecordedToolCall,
    Recording,
)
from tests.llm.simulation.simulator import (
    SimulatorParseError,
    ToolCallSimulator,
    _extract_json_block,
)


# ----- helpers ---------------------------------------------------------------


def _rec() -> Recording:
    return Recording(
        system_prompt="s",
        user_prompt="u",
        tool_calls=[
            RecordedToolCall(
                id="r1",
                name="kubectl_get_pods",
                arguments={"namespace": "checkout"},
                response=StructuredToolResult(
                    status=StructuredToolResultStatus.SUCCESS,
                    data="POD A   Running",
                ),
            ),
            RecordedToolCall(
                id="r2",
                name="kubectl_logs",
                arguments={"pod": "checkout-7f"},
                response=StructuredToolResult(
                    status=StructuredToolResultStatus.SUCCESS,
                    data="connection refused redis",
                ),
            ),
        ],
        ground_truth_diagnosis="redis is OOMKilled",
    )


def _make_response(content: str, tool_calls=None):
    """Build a minimal ModelResponse-like object with one assistant choice."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _tc(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


# ----- _extract_json_block ---------------------------------------------------


def test_extract_json_block_fenced() -> None:
    text = 'reasoning text\n```json\n{"status": "success", "data": "ok"}\n```\n'
    payload = _extract_json_block(text)
    assert payload == {"status": "success", "data": "ok"}


def test_extract_json_block_unfenced() -> None:
    text = 'sure! here it is: {"status": "no_data"} '
    payload = _extract_json_block(text)
    assert payload == {"status": "no_data"}


def test_extract_json_block_returns_none_on_garbage() -> None:
    assert _extract_json_block("nothing parseable here") is None


# ----- meta-tool dispatch ----------------------------------------------------


def test_meta_fetch_returns_recorded_response() -> None:
    sim = ToolCallSimulator(_rec(), llm=MagicMock())
    out = sim._meta_fetch("r1")
    parsed = json.loads(out)
    assert parsed["name"] == "kubectl_get_pods"
    assert parsed["response"]["status"] == "success"
    assert parsed["response"]["data"] == "POD A   Running"


def test_meta_fetch_unknown_id() -> None:
    sim = ToolCallSimulator(_rec(), llm=MagicMock())
    out = sim._meta_fetch("rXX")
    parsed = json.loads(out)
    assert "error" in parsed
    assert "available_ids" in parsed
    assert "r1" in parsed["available_ids"]


def test_meta_search_substring_matches_name_and_args() -> None:
    sim = ToolCallSimulator(_rec(), llm=MagicMock())
    out = sim._meta_search("logs")
    matches = json.loads(out)["matches"]
    assert any(m["id"] == "r2" for m in matches)

    out = sim._meta_search("checkout")
    matches = json.loads(out)["matches"]
    # r1 args contain "checkout"; r2 args contain "checkout-7f"
    ids = {m["id"] for m in matches}
    assert {"r1", "r2"}.issubset(ids)


def test_meta_search_empty_query() -> None:
    sim = ToolCallSimulator(_rec(), llm=MagicMock())
    out = sim._meta_search("")
    assert json.loads(out)["matches"] == []


# ----- simulate() ------------------------------------------------------------


def test_simulate_returns_parsed_result_one_turn() -> None:
    """Simulator-LLM emits a final answer immediately; no meta-tools used."""
    llm = MagicMock()
    llm.completion.return_value = _make_response(
        '```json\n{"status": "success", "data": "POD A Running"}\n```'
    )
    sim = ToolCallSimulator(_rec(), llm=llm)
    out = sim.simulate("kubectl_get_pods", {"namespace": "checkout"})
    assert out.status == StructuredToolResultStatus.SUCCESS
    assert out.data == "POD A Running"
    assert llm.completion.call_count == 1


def test_simulate_with_meta_tool_then_final() -> None:
    """First turn: simulator calls fetch_recorded_response.
    Second turn: simulator emits final JSON."""
    llm = MagicMock()
    fetch_call = _tc("call1", "fetch_recorded_response", {"call_id": "r1"})
    llm.completion.side_effect = [
        _make_response("", tool_calls=[fetch_call]),
        _make_response('```json\n{"status": "success", "data": "POD A Running"}\n```'),
    ]
    sim = ToolCallSimulator(_rec(), llm=llm)
    out = sim.simulate("kubectl_get_pods", {"namespace": "checkout"})
    assert out.status == StructuredToolResultStatus.SUCCESS
    assert out.data == "POD A Running"
    assert llm.completion.call_count == 2

    # The second completion call should have received the assistant message
    # plus a tool result message containing the recorded response.
    second_call_messages = llm.completion.call_args_list[1].kwargs["messages"]
    tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "POD A   Running" in tool_msg["content"]


def test_simulate_returns_error_on_bad_status() -> None:
    llm = MagicMock()
    llm.completion.return_value = _make_response(
        '```json\n{"status": "weird_status"}\n```'
    )
    sim = ToolCallSimulator(_rec(), llm=llm)
    out = sim.simulate("kubectl_get_pods", {})
    # SimulatorParseError caught and surfaced as ERROR result.
    assert out.status == StructuredToolResultStatus.ERROR
    assert out.error and "invalid status" in out.error


def test_simulate_returns_error_when_no_final_within_iterations() -> None:
    llm = MagicMock()
    fetch_call = _tc("c", "fetch_recorded_response", {"call_id": "r1"})
    # Always emits a tool call -> never converges.
    llm.completion.return_value = _make_response("", tool_calls=[fetch_call])
    sim = ToolCallSimulator(_rec(), llm=llm, max_meta_iterations=2)
    out = sim.simulate("kubectl_get_pods", {})
    assert out.status == StructuredToolResultStatus.ERROR
    assert "did not emit a final answer" in (out.error or "")


def test_simulate_handles_no_data_status() -> None:
    llm = MagicMock()
    llm.completion.return_value = _make_response(
        '```json\n{"status": "no_data"}\n```'
    )
    sim = ToolCallSimulator(_rec(), llm=llm)
    out = sim.simulate("prom_query", {"q": "rate(...)"})
    assert out.status == StructuredToolResultStatus.NO_DATA


def test_simulate_lenient_falls_back_to_plain_text() -> None:
    """Small models often emit markdown instead of the JSON block. The
    parser must treat that as a plain SUCCESS result so the agent can keep
    investigating, rather than silently returning ERROR."""
    llm = MagicMock()
    llm.completion.return_value = _make_response(
        "Here are the pods:\n\nNAME   STATUS\ncheckout-7f   CrashLoopBackOff"
    )
    sim = ToolCallSimulator(_rec(), llm=llm)
    out = sim.simulate("kubectl_get_pods", {"namespace": "checkout"})
    assert out.status == StructuredToolResultStatus.SUCCESS
    assert "CrashLoopBackOff" in (out.data or "")


def test_simulate_empty_text_is_error() -> None:
    """Truly empty final messages should still be ERROR — that genuinely
    indicates the simulator failed."""
    llm = MagicMock()
    llm.completion.return_value = _make_response("")
    sim = ToolCallSimulator(_rec(), llm=llm)
    out = sim.simulate("kubectl_get_pods", {})
    assert out.status == StructuredToolResultStatus.ERROR
