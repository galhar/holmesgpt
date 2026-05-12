"""Recording schema — the JSON contract between captured customer sessions and
the simulator harness.

A recording is a single JSON file describing one diagnostic episode:
- the system prompt Holmes ran with,
- the user's question (or alert text),
- the tool calls Holmes ran during diagnosis with their real responses,
- the human-confirmed final diagnosis (the "ground truth").

The simulator (see ``simulator.py``) is given the recording and uses it to
fabricate realistic responses for tool calls the agent under test makes during
the simulated run. The agent under test never sees recorded responses
directly — only the simulator does, and only when it chooses to fetch them.

Schema versioning:
    schema_version field exists so older recordings keep parsing as the schema
    evolves. Bump the major version on breaking changes; add fields with
    sensible defaults otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from holmes.core.tools import StructuredToolResult


class RecordedToolCall(BaseModel):
    """One real tool call captured from a customer's diagnostic session.

    The ``response`` field embeds HolmesGPT's existing ``StructuredToolResult``
    (defined in ``holmes/core/tools.py``) verbatim so that recordings can be
    produced by serializing real results without translation.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description=(
            "Stable short identifier (e.g. 'r1', 'r2'). Used by the simulator's "
            "meta-tools to address this call. Must be unique within a recording."
        ),
    )
    name: str = Field(..., description="Tool name as registered in HolmesGPT.")
    arguments: dict = Field(
        default_factory=dict,
        description="The params the agent passed when this call ran in production.",
    )
    invocation: Optional[str] = Field(
        default=None,
        description=(
            "Optional human-readable command line, used purely for display in "
            "logs and the simulator's listing. Not interpreted."
        ),
    )
    response: StructuredToolResult = Field(
        ...,
        description="The real recorded tool response.",
    )


class Recording(BaseModel):
    """A captured diagnostic session, usable as a simulated-environment scenario.

    The fields are deliberately minimal — Simia-RL keeps its scenario format to
    two keys (``system`` and ``conversations``); we add ``user_prompt`` and
    ``ground_truth_diagnosis`` because HolmesGPT's loop is single-question
    rather than free conversation, and we need the gt for both the simulator's
    system prompt and the eval's pass/fail judge.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="holmes.sim:v1",
        description=(
            "Schema version. Older recordings continue to parse as long as "
            "newly added fields have defaults."
        ),
    )
    system_prompt: str = Field(
        ...,
        description=(
            "The system prompt Holmes ran with at recording time. Used as the "
            "agent-under-test's system prompt during replay for maximum fidelity."
        ),
    )
    user_prompt: str = Field(
        ..., description="The customer's question / alert text."
    )
    tool_calls: list[RecordedToolCall] = Field(
        default_factory=list,
        description=(
            "Tool calls the agent ran during diagnosis, in chronological order. "
            "The simulator sees their (id, name, arguments) up front and can "
            "fetch responses on demand via meta-tools."
        ),
    )
    ground_truth_diagnosis: str = Field(
        ...,
        description=(
            "Human-confirmed final diagnosis. Used in the simulator's system "
            "prompt to guide tool-response fabrication, and as the target for "
            "the eval's correctness judge. Never shown to the agent under test."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form metadata: customer, ticket id, original model, capture "
            "date, etc. Not interpreted by the harness."
        ),
    )

    def get_recorded_call(self, call_id: str) -> Optional[RecordedToolCall]:
        """Return the recorded call with the given id, or None."""
        for c in self.tool_calls:
            if c.id == call_id:
                return c
        return None


def load_recording(path: Union[str, Path]) -> Recording:
    """Load a recording from a JSON file."""
    text = Path(path).read_text(encoding="utf-8")
    return Recording.model_validate_json(text)


def save_recording(recording: Recording, path: Union[str, Path]) -> None:
    """Write a recording to a JSON file (pretty-printed, UTF-8)."""
    Path(path).write_text(
        recording.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )


def recording_to_summary(recording: Recording) -> str:
    """Human-readable one-paragraph summary, for logging and CLI listings."""
    return (
        f"Recording schema={recording.schema_version} "
        f"with {len(recording.tool_calls)} tool calls; "
        f"prompt={recording.user_prompt[:60]!r}..."
    )


def list_call_signatures(recording: Recording) -> str:
    """Render the recorded calls as `[id] name(args_json)` lines.

    Used inside the simulator's system prompt so the simulator-LLM can see
    what's available without paying the token cost of the responses up front.
    """
    lines: list[str] = []
    for c in recording.tool_calls:
        try:
            args_json = json.dumps(c.arguments, ensure_ascii=False)
        except Exception:
            args_json = str(c.arguments)
        lines.append(f"  [{c.id}] {c.name}({args_json})")
    return "\n".join(lines) if lines else "  (no recorded tool calls)"
