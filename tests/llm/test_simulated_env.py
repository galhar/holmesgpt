# type: ignore
"""Eval entry-point for simulated-environment tests.

Parallel to ``test_ask_holmes.py`` but instead of raising real infrastructure,
every tool call is intercepted by an LLM-as-simulator (see
``tests/llm/simulation/``).

Test discovery
--------------
Each fixture lives in ``tests/llm/fixtures/test_simulated_env/<id>/`` and
contains:
    - ``recording.json`` — the captured customer session (see
      ``tests/llm/simulation/recording.py``).
    - ``test_case.yaml``  — minimal eval config: ``recording: recording.json``
      plus optional ``expected_output``, ``tags``, ``toolsets``, ``skip``, ...

Pass/fail
---------
Reuses the same ``evaluate_correctness`` judge as ``test_ask_holmes``. By
default the judge compares the agent's final answer against the recording's
``ground_truth_diagnosis``. If ``expected_output`` is set in the test_case
YAML it overrides the gt for scoring (useful when the gt is verbose and you
want to assert on specific bullets).

This file does NOT request the ``shared_test_infrastructure`` fixture — there
is no infrastructure to raise. It also does NOT support port forwards or
``before_test``/``after_test`` hooks (those fields are simply ignored).
"""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from os import path
from pathlib import Path
from typing import List, Optional, Union

import pytest
import yaml
from pydantic import ConfigDict, ValidationError

from holmes.core.prompt import build_initial_ask_messages
from holmes.core.tool_calling_llm import LLMResult, ToolCallingLLM
from holmes.core.tools_utils.filesystem_result_storage import tool_result_storage
from holmes.core.tracing import SpanType, TracingFactory
from holmes.plugins.skills.skill_loader import load_skill_catalog

from tests.llm.simulation.interceptor import build_simulated_tool_executor
from tests.llm.simulation.recording import Recording, load_recording
from tests.llm.simulation.simulator import ToolCallSimulator
from tests.llm.utils.classifiers import evaluate_correctness
from tests.llm.utils.commands import set_test_env_vars
from tests.llm.utils.env_config import EnvConfig, get_env_configs
from tests.llm.utils.iteration_utils import add_id_and_tags_to_test_case
from tests.llm.utils.property_manager import (
    set_initial_properties,
    set_trace_properties,
)
from tests.llm.utils.test_case_utils import (
    AskHolmesTestCase,
    create_eval_llm,
    get_models,
)
from tests.llm.utils.test_toolset import TestToolsetManager

TEST_CASES_FOLDER = Path(
    path.abspath(path.join(path.dirname(__file__), "fixtures", "test_simulated_env"))
)


class SimulatedEnvTestCase(AskHolmesTestCase):
    """Eval test case backed by a simulated environment.

    Inherits from ``AskHolmesTestCase`` (so all eval bookkeeping — tags,
    toolsets, skip flags, etc. — works identically), and adds the ``recording``
    field pointing at a JSON file relative to the test case folder.

    The agent under test ALWAYS goes through Holmes's normal
    ``build_initial_ask_messages`` path (same as ``test_ask_holmes``) so that
    the system prompt, tool schemas, and skill catalog are identical to a
    real run. ``recording.system_prompt`` is treated as captured metadata
    (kept for documentation/audit) rather than as the prompt the agent runs
    with — using it would diverge the simulated harness from the production
    agent in subtle ways.

    ``expected_output`` and ``user_prompt`` are made optional here because
    both can be derived from the recording if absent.
    """

    model_config = ConfigDict(extra="forbid")

    recording: str  # Path relative to the test case folder.
    expected_output: Optional[Union[str, List[str]]] = None  # type: ignore[assignment]
    user_prompt: Optional[Union[str, List[str]]] = None  # type: ignore[assignment]


def _load_simulated_env_test_cases() -> list:
    """Discover and load every fixture under ``test_simulated_env``."""
    if not TEST_CASES_FOLDER.exists():
        return []

    cases = []
    for entry in sorted(os.listdir(TEST_CASES_FOLDER)):
        if entry.startswith("."):
            continue
        folder = TEST_CASES_FOLDER / entry
        cfg_path = folder / "test_case.yaml"
        if not folder.is_dir() or not cfg_path.exists():
            continue
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        config["id"] = entry
        config["folder"] = str(folder)
        try:
            tc = SimulatedEnvTestCase(**config)
        except ValidationError as e:
            raise pytest.UsageError(
                f"Invalid test_case.yaml in {folder}: {e}"
            ) from e
        cases.append(add_id_and_tags_to_test_case(tc))
    return cases


def _get_env_config_ids():
    return [ec.name for ec in get_env_configs()]


def _resolve_simulator_model(default_model: str) -> str:
    """Pick the model for the simulator LLM. Falls back to the agent's model."""
    return os.environ.get("SIMULATOR_MODEL") or default_model


def _build_messages(
    recording: Recording,
    test_case: SimulatedEnvTestCase,
    tool_executor,
    additional_system_prompt: Optional[str],
) -> list:
    """Construct the agent-under-test's initial conversation.

    Mirrors the CLI path in ``test_ask_holmes.py:228-235`` so the agent gets
    the same system prompt, tool descriptions, and skill catalog it would
    see in production — only the *tool execution* is faked.
    """
    user_prompt = test_case.user_prompt or recording.user_prompt
    skills = load_skill_catalog(custom_skill_paths=[Path(test_case.folder)])
    return build_initial_ask_messages(
        initial_user_prompt=user_prompt,
        file_paths=None,
        tool_executor=tool_executor,
        skills=skills,
        system_prompt_additions=additional_system_prompt,
    )


@pytest.mark.llm
@pytest.mark.simulated
@pytest.mark.parametrize("env_config", get_env_configs(), ids=_get_env_config_ids())
@pytest.mark.parametrize("model", get_models())
@pytest.mark.parametrize("test_case", _load_simulated_env_test_cases())
def test_simulated_env(
    env_config: EnvConfig,
    model: str,
    test_case: SimulatedEnvTestCase,
    caplog,
    request,
    additional_system_prompt,
):
    """Run a single simulated-environment eval.

    No ``shared_test_infrastructure`` is requested here — simulated tests
    don't raise real services. ``before_test``/``after_test``/``port_forwards``
    in ``test_case.yaml`` are ignored (the harness does not invoke them).
    """
    set_initial_properties(request, test_case, model, env_config)

    if test_case.skip:
        pytest.skip(test_case.skip_reason or "Test skipped")

    tracer = TracingFactory.create_tracer("braintrust")
    metadata = {
        "model": model,
        "env_config": env_config.name,
        "harness": "simulated_env",
    }
    tracer.start_experiment(additional_metadata=metadata)

    result: Optional[LLMResult] = None

    with tracer.start_trace(
        name=f"{test_case.id}[{model}][{env_config.name}]", span_type=SpanType.EVAL
    ) as eval_span:
        set_trace_properties(request, eval_span)

        with ExitStack() as stack:
            stack.enter_context(set_test_env_vars(test_case))

            recording_path = Path(test_case.folder) / test_case.recording
            recording = load_recording(recording_path)

            simulator_model = _resolve_simulator_model(model)
            simulator_llm = create_eval_llm(model=simulator_model, tracer=tracer)
            simulator = ToolCallSimulator(recording=recording, llm=simulator_llm)

            with eval_span.start_span(
                "Initialize Toolsets", type=SpanType.TASK.value
            ) as toolset_span:
                toolset_manager = TestToolsetManager(
                    test_case_folder=test_case.folder,
                    allow_toolset_failures=getattr(
                        test_case, "allow_toolset_failures", False
                    ),
                    toolsets_config_path=getattr(
                        test_case, "toolsets_config_path", None
                    ),
                )
                from holmes.core.tools_utils.tool_executor import ToolExecutor

                real_executor = ToolExecutor(toolset_manager.toolsets)
                tool_executor = build_simulated_tool_executor(
                    real_executor, simulator
                )
                enabled = [t.name for t in tool_executor.enabled_toolsets]
                tool_names = sorted(tool_executor.tools_by_name.keys())
                print(
                    f"\n🤖 SIMULATED-ENV TOOLSETS ({len(enabled)}): "
                    + ", ".join(enabled)
                )
                # Sanity dump: agent under test must have access to the FULL
                # Holmes tool surface (each tool wrapped to delegate to the
                # simulator). If this list is short, the wrapping is wrong.
                print(
                    f"\n🛠️  TOOLS PASSED TO AGENT ({len(tool_names)} total):"
                )
                for i, n in enumerate(tool_names, 1):
                    print(f"   {i:>3}. {n}")
                toolset_span.log(
                    metadata={
                        "toolset_names": enabled,
                        "tool_count": len(tool_names),
                        "tool_names": tool_names,
                    }
                )

            with tool_result_storage() as tool_results_dir:
                ai = ToolCallingLLM(
                    tool_executor=tool_executor,
                    max_steps=100,
                    llm=create_eval_llm(model=model, tracer=tracer),
                    tool_results_dir=tool_results_dir,
                )
                messages = _build_messages(
                    recording=recording,
                    test_case=test_case,
                    tool_executor=tool_executor,
                    additional_system_prompt=additional_system_prompt,
                )
                print(
                    f"\n📝 AGENT MESSAGES: {len(messages)} initial messages "
                    f"(system={sum(1 for m in messages if m['role'] == 'system')}, "
                    f"user={sum(1 for m in messages if m['role'] == 'user')})"
                )
                # Confirm what ToolCallingLLM will actually pass to the LLM via
                # tools=... parameter. This is the LLM's tool surface.
                openai_tools = tool_executor.get_all_tools_openai_format(
                    include_restricted=False, user_id=None
                )
                openai_tool_names = sorted(
                    t["function"]["name"] for t in openai_tools
                )
                print(
                    f"\n🔌 OPENAI-FORMAT TOOLS PASSED TO LLM "
                    f"({len(openai_tools)}): {', '.join(openai_tool_names[:8])}"
                    + (
                        f", … (+{len(openai_tool_names) - 8} more)"
                        if len(openai_tool_names) > 8
                        else ""
                    )
                )

                with tracer.start_trace(
                    "Holmes Run (simulated)", span_type=SpanType.TASK
                ) as llm_span:
                    start = time.time()
                    result = ai.call(messages=messages, trace_span=llm_span)
                    holmes_duration = time.time() - start
                    eval_span.log(metadata={"holmes_duration": holmes_duration})

                print(f"\n✅ AGENT FINAL ANSWER:\n{result.result}\n")
                print(
                    f"   tool_calls: {len(result.tool_calls or [])}, "
                    f"num_llm_calls: {result.num_llm_calls}, "
                    f"duration: {holmes_duration:.1f}s"
                )

    output = result.result if result else None

    expected = test_case.expected_output or recording.ground_truth_diagnosis

    eval_obj = evaluate_correctness(
        expected_elements=expected,
        output=output or "",
        parent_span=eval_span,
        caplog=caplog,
    )
    score = int(eval_obj.score) if eval_obj.score is not None else 0

    assert score == 1, (
        f"Test {test_case.id} failed (score: {eval_obj.score})\n"
        f"Actual: {output}\n"
        f"Expected:\n  {expected}"
    )
