"""Tests for the Recording schema (load/save round-trip and rendering helpers)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus
from tests.llm.simulation.recording import (
    RecordedToolCall,
    Recording,
    list_call_signatures,
    load_recording,
    recording_to_summary,
    save_recording,
)


def _sample_recording() -> Recording:
    return Recording(
        system_prompt="sysprompt",
        user_prompt="why is checkout failing?",
        tool_calls=[
            RecordedToolCall(
                id="r1",
                name="kubectl_get_pods",
                arguments={"namespace": "checkout"},
                response=StructuredToolResult(
                    status=StructuredToolResultStatus.SUCCESS,
                    data="NAME ... CrashLoopBackOff",
                ),
            ),
            RecordedToolCall(
                id="r2",
                name="prom_query",
                arguments={"q": "rate(http_5xx[5m])"},
                response=StructuredToolResult(
                    status=StructuredToolResultStatus.NO_DATA,
                    data=None,
                ),
            ),
        ],
        ground_truth_diagnosis="redis OOMKilled",
    )


def test_roundtrip(tmp_path: Path) -> None:
    rec = _sample_recording()
    path = tmp_path / "rec.json"
    save_recording(rec, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == "holmes.sim:v1"
    assert raw["tool_calls"][0]["response"]["status"] == "success"

    loaded = load_recording(path)
    assert loaded == rec


def test_get_recorded_call_returns_match() -> None:
    rec = _sample_recording()
    assert rec.get_recorded_call("r1").name == "kubectl_get_pods"
    assert rec.get_recorded_call("does_not_exist") is None


def test_list_call_signatures_includes_id_name_args() -> None:
    rec = _sample_recording()
    rendered = list_call_signatures(rec)
    assert "[r1] kubectl_get_pods" in rendered
    assert '"namespace": "checkout"' in rendered
    assert "[r2] prom_query" in rendered


def test_list_call_signatures_handles_empty() -> None:
    rec = Recording(
        system_prompt="x",
        user_prompt="y",
        tool_calls=[],
        ground_truth_diagnosis="z",
    )
    assert "no recorded tool calls" in list_call_signatures(rec)


def test_recording_to_summary_includes_count() -> None:
    rec = _sample_recording()
    s = recording_to_summary(rec)
    assert "2 tool calls" in s


def test_extra_fields_rejected() -> None:
    with pytest.raises(Exception):
        Recording(
            system_prompt="s",
            user_prompt="u",
            tool_calls=[],
            ground_truth_diagnosis="g",
            unexpected_field="boom",
        )
