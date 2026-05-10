"""LLM-as-simulator. Given an agent's tool call, produces a fabricated
``StructuredToolResult`` consistent with a recording and its ground truth.

Architecture
------------
A ``ToolCallSimulator`` wraps:
  * a ``Recording`` (the scenario), and
  * an ``LLM`` (a ``holmes.core.llm.DefaultLLM`` instance — same LiteLLM client
    Holmes itself uses).

When the agent under test calls a tool, ``simulate(tool_name, tool_args)`` runs
a small inner ReAct-style loop where the simulator-LLM has access to two
meta-tools (defined in this file as plain Python dispatch — they don't go
through Holmes's tool registry):

    fetch_recorded_response(call_id)   -> returns a recorded StructuredToolResult
    search_recordings(query)           -> substring search; returns matching ids

After at most ``max_meta_iterations`` turns, the simulator-LLM must emit a
final assistant message containing one fenced ``json`` block with a
``StructuredToolResult`` payload (status/data/error). We parse and return it.

Why this design
---------------
- We DON'T reuse ``ToolCallingLLM`` because the simulator's loop is
  intentionally tiny and decoupled from Holmes's. Mixing concerns would
  couple the simulator to Holmes's safeguards, compaction, and tracing — none
  of which apply here.
- We DON'T stuff every recorded response into the system prompt. Real customer
  sessions can have 30+ tool calls with multi-MB responses; up-front
  concatenation explodes context. Lazy fetch keeps the simulator focused.
- Why ``temperature=0``? Determinism. Inputs are: gt + recordings + agent's
  call. We want the same simulated answer every time so eval pass/fail is
  reproducible across runs.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template

from holmes.core.llm import LLM
from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus

from tests.llm.simulation.recording import (
    Recording,
    list_call_signatures,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "simulator_system.jinja2"


# OpenAI-format tool schemas for the simulator's meta-tools. These are passed
# directly to ``llm.completion(tools=...)``; we dispatch tool calls in
# ``_execute_meta_tool`` below by matching on ``function.name``.
_META_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_recorded_response",
            "description": (
                "Fetch the full StructuredToolResult of a recorded tool call by "
                "its id (e.g. 'r1'). Use when the agent's call matches a "
                "recorded one and you want to return the real response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "call_id": {
                        "type": "string",
                        "description": "The recorded call id, as listed in the system prompt.",
                    }
                },
                "required": ["call_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_recordings",
            "description": (
                "Substring-search the recorded tool calls' names and argument "
                "JSON. Returns a list of matching call ids. Useful when you "
                "remember a tool name fragment but not the exact id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive substring to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


class SimulatorParseError(RuntimeError):
    """Raised when the simulator-LLM does not produce a parseable
    StructuredToolResult JSON block within the iteration cap."""


class ToolCallSimulator:
    """Generates simulated tool responses from a recording + an LLM.

    Public surface is intentionally one method:

        sim = ToolCallSimulator(recording, llm)
        result: StructuredToolResult = sim.simulate("kubectl_get_pods", {"ns": "x"})

    Internals are exposed as protected methods only for unit testing.
    """

    def __init__(
        self,
        recording: Recording,
        llm: LLM,
        max_meta_iterations: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self.recording = recording
        self.llm = llm
        self.max_meta_iterations = max_meta_iterations
        self.temperature = temperature
        self._system_prompt_template = Template(
            _PROMPT_PATH.read_text(encoding="utf-8")
        )
        # Indexed view of the recording for O(1) meta-tool dispatch.
        self._calls_by_id: dict[str, int] = {
            c.id: i for i, c in enumerate(recording.tool_calls)
        }

    # ------------------------------------------------------------------ public

    def simulate(self, tool_name: str, tool_args: dict) -> StructuredToolResult:
        """Run the inner loop and return a fabricated tool result.

        Falls back to a SUCCESS result with the parse error in ``data`` if the
        simulator-LLM fails to produce a valid JSON block within the iteration
        cap — that way a single bad simulator turn doesn't crash the whole
        eval, it just gives the agent under test a degenerate response.
        """
        try:
            return self._run_inner_loop(tool_name, tool_args)
        except SimulatorParseError as e:
            logger.warning(
                "Simulator failed to produce valid JSON for %s(%s): %s",
                tool_name,
                tool_args,
                e,
            )
            return StructuredToolResult(
                status=StructuredToolResultStatus.ERROR,
                error=f"simulator: {e}",
                data=None,
                params=tool_args,
                invocation=f"<simulated> {tool_name}",
            )

    # ----------------------------------------------------------------- helpers

    def _build_system_prompt(self, tool_name: str, tool_args: dict) -> str:
        try:
            tool_args_json = json.dumps(tool_args, ensure_ascii=False)
        except Exception:
            tool_args_json = str(tool_args)
        return self._system_prompt_template.render(
            user_prompt=self.recording.user_prompt,
            ground_truth_diagnosis=self.recording.ground_truth_diagnosis,
            recorded_call_signatures=list_call_signatures(self.recording),
            tool_name=tool_name,
            tool_args_json=tool_args_json,
        )

    def _run_inner_loop(
        self, tool_name: str, tool_args: dict
    ) -> StructuredToolResult:
        system_prompt = self._build_system_prompt(tool_name, tool_args)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        for turn_idx in range(1, self.max_meta_iterations + 1):
            logger.info(
                "[simulator] turn %d/%d for agent_call=%s",
                turn_idx,
                self.max_meta_iterations,
                tool_name,
            )
            response = self.llm.completion(
                messages=messages,
                tools=_META_TOOLS,
                tool_choice="auto",
                temperature=self.temperature,
                drop_params=True,
            )

            choice = response.choices[0]  # type: ignore[union-attr]
            assistant_msg = choice.message
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            finish_reason = getattr(choice, "finish_reason", "?")

            if tool_calls:
                logger.info(
                    "[simulator]   turn %d emitted %d meta-tool call(s); finish=%s",
                    turn_idx,
                    len(tool_calls),
                    finish_reason,
                )
                # Replay assistant message with its tool_calls so subsequent
                # tool-role messages have a parent to attach to.
                messages.append(_assistant_message_to_dict(assistant_msg))
                for tc in tool_calls:
                    name = getattr(tc.function, "name", "?")
                    args_preview = (tc.function.arguments or "")[:120]
                    logger.info(
                        "[simulator]     -> %s(%s)", name, args_preview
                    )
                    result = self._execute_meta_tool(tc)
                    logger.info(
                        "[simulator]     <- %d chars from %s",
                        len(result),
                        name,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": result,
                        }
                    )
                continue

            # No tool calls: the assistant gave a final answer. Parse it.
            text = (assistant_msg.content or "").strip()
            logger.info(
                "[simulator]   turn %d emitted final answer (%d chars); finish=%s",
                turn_idx,
                len(text),
                finish_reason,
            )
            return self._parse_final_answer(text, tool_args, tool_name)

        raise SimulatorParseError(
            f"simulator did not emit a final answer within {self.max_meta_iterations} turns"
        )

    def _execute_meta_tool(self, tc: Any) -> str:
        """Dispatch one of the simulator's meta-tool calls to local Python.

        Returns the tool result as a string (which is what LiteLLM expects in
        the tool message ``content`` field).
        """
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            return f'{{"error": "could not parse tool arguments: {tc.function.arguments!r}"}}'

        if name == "fetch_recorded_response":
            return self._meta_fetch(args.get("call_id", ""))
        if name == "search_recordings":
            return self._meta_search(args.get("query", ""))
        return f'{{"error": "unknown meta-tool: {name}"}}'

    def _meta_fetch(self, call_id: str) -> str:
        idx = self._calls_by_id.get(call_id)
        if idx is None:
            return json.dumps(
                {
                    "error": f"no recorded call with id={call_id!r}",
                    "available_ids": list(self._calls_by_id.keys()),
                }
            )
        call = self.recording.tool_calls[idx]
        return json.dumps(
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
                "response": call.response.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )

    def _meta_search(self, query: str) -> str:
        if not query:
            return json.dumps({"matches": []})
        q = query.lower()
        matches: list[dict[str, str]] = []
        for c in self.recording.tool_calls:
            try:
                args_text = json.dumps(c.arguments, ensure_ascii=False).lower()
            except Exception:
                args_text = str(c.arguments).lower()
            if q in c.name.lower() or q in args_text:
                matches.append({"id": c.id, "name": c.name})
        return json.dumps({"matches": matches}, ensure_ascii=False)

    def _parse_final_answer(
        self, text: str, tool_args: dict, tool_name: str
    ) -> StructuredToolResult:
        """Parse the simulator-LLM's final assistant message into a result.

        Small open models (Hermes-class via the Ollama OpenAI shim, etc.)
        often emit plausible tool output as plain markdown rather than the
        JSON block we ask for. So we accept either:

          1. A fenced ``json`` block with status/data/error fields (preferred,
             needed to express ERROR/NO_DATA statuses).
          2. Any other non-empty text — treated as the ``data`` field of a
             SUCCESS result. This loses the ability for the simulator to
             signal an error on this turn, but ensures the agent sees a
             non-empty response and can keep investigating.

        Empty text on a final turn is the only condition that yields an
        ERROR — that genuinely indicates the simulator failed.
        """
        payload = _extract_json_block(text)

        if payload is not None:
            status_str = (payload.get("status") or "success").lower()
            try:
                status = StructuredToolResultStatus(status_str)
            except ValueError:
                raise SimulatorParseError(
                    f"invalid status {status_str!r} (must be success/error/no_data)"
                )
            return StructuredToolResult(
                status=status,
                data=payload.get("data"),
                error=payload.get("error"),
                return_code=payload.get("return_code"),
                params=tool_args,
                invocation=f"<simulated> {tool_name}",
            )

        if text.strip():
            logger.info(
                "[simulator] no JSON block in final answer — treating "
                "%d chars as plain SUCCESS data",
                len(text),
            )
            return StructuredToolResult(
                status=StructuredToolResultStatus.SUCCESS,
                data=text.strip(),
                params=tool_args,
                invocation=f"<simulated> {tool_name}",
            )

        raise SimulatorParseError(
            "simulator emitted an empty final message"
        )


# --------------------------------------------------------------- module utils


def _extract_json_block(text: str) -> Optional[dict]:
    """Return the first parseable JSON object found in ``text``.

    Tries fenced ```json blocks first, then falls back to the first balanced
    ``{...}`` substring. Returns None if nothing parseable is found.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: greedy match of the first {...} that parses.
    for m in re.finditer(r"\{.*\}", text, re.DOTALL):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _assistant_message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a LiteLLM assistant message (with tool_calls) to a plain dict
    suitable for re-feeding into the next ``completion`` call."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return out
