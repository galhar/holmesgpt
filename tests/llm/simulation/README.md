# Simulated-environment eval harness for HolmesGPT

This module lets you eval HolmesGPT on a **fake world** instead of real
infrastructure. Every tool call the agent makes is intercepted by an
**LLM-as-simulator** that fabricates a realistic response, guided by a
captured "recording" of a real customer's diagnostic session.

It is the cousin of `test_ask_holmes`:

|                          | `test_ask_holmes`                      | `test_simulated_env` (this module)                   |
|--------------------------|----------------------------------------|------------------------------------------------------|
| Backend                  | Real (KIND, real APIs, real DBs)       | Faked by LLM-as-simulator                            |
| `before_test` / `after_test` | Required for most tests           | Not used                                             |
| Setup time               | Seconds–minutes per test               | Zero                                                 |
| Cost                     | Cluster CPU + LLM tokens               | Simulator tokens + agent tokens                      |
| Best for                 | Validating against real APIs           | Multi-system scenarios that are hard to reproduce locally |

## At a glance

```
agent under test         SimulatedTool wrapper          ToolCallSimulator
(ToolCallingLLM)  ──►   (overrides _invoke)      ──►   (LLM with meta-tools)
                              │
                              │ Tool.invoke() also runs the
                              │ existing transformer chain
                              │ (incl. llm_summarize) on the
                              │ simulator's output, exactly as
                              │ in production
                              ▼
                       agent sees a normal tool result
```

## Recording schema

A recording is a single JSON file with these fields (full Pydantic schema in
[`recording.py`](recording.py)):

| Field                       | Type                       | Meaning |
|-----------------------------|----------------------------|---------|
| `schema_version`            | `string` (default `"holmes.sim:v1"`) | Bumped on breaking changes. |
| `system_prompt`             | `string`                   | The system prompt Holmes ran with at recording time. Used as the agent-under-test's system prompt by default for fidelity. |
| `user_prompt`               | `string`                   | The customer's question or alert text. |
| `tool_calls`                | `list[RecordedToolCall]`   | Tool calls Holmes made during diagnosis, with their **real** responses. |
| `ground_truth_diagnosis`    | `string`                   | Human-confirmed final diagnosis. Used in the simulator's system prompt and as the eval's correctness target. **Never shown to the agent under test.** |
| `metadata`                  | `dict`                     | Free-form (customer, ticket id, original model, etc.). |

`RecordedToolCall` fields:

| Field         | Type                   | Meaning |
|---------------|------------------------|---------|
| `id`          | `string`               | Stable short id (`r1`, `r2`, …). Used by the simulator's meta-tools to address this call. Must be unique within a recording. |
| `name`        | `string`               | Tool name as registered in HolmesGPT (e.g. `kubectl_get_pods`). |
| `arguments`   | `dict`                 | Arguments the agent passed when this call ran in production. |
| `invocation`  | `string?`              | Optional human-readable command line for display. |
| `response`    | `StructuredToolResult` | The real recorded response. See below. |

`StructuredToolResult` (defined in `holmes/core/tools.py`):

| Field           | Type     | Notes |
|-----------------|----------|-------|
| `status`        | enum     | `success` \| `error` \| `no_data` \| `approval_required` \| `frontend_pause` |
| `data`          | any      | The tool output. String, dict, list, or null. |
| `error`         | string?  | Error message when `status == "error"`. |
| `return_code`   | int?     | Optional exit code for shell-like tools. |
| `params`        | dict?    | The params used (echoed back). |
| `invocation`    | string?  | Human-readable command. |
| `images`        | list?    | Optional base64-encoded images. |
| `url`           | string?  | Optional source URL. |
| `elapsed_seconds` | float? | Execution duration. |

## Authoring a recording by hand

Minimum viable example — copy this and fill in:

```json
{
  "system_prompt": "You are HolmesGPT, an SRE assistant ...",
  "user_prompt": "Why are checkout requests failing?",
  "tool_calls": [
    {
      "id": "r1",
      "name": "kubectl_get_pods",
      "arguments": {"namespace": "checkout"},
      "response": {
        "status": "success",
        "data": "NAME   READY  STATUS\ncheckout-7f...  0/1  CrashLoopBackOff"
      }
    }
  ],
  "ground_truth_diagnosis": "checkout-api crashlooping; root cause: redis OOMKilled."
}
```

`schema_version` defaults to `holmes.sim:v1` if omitted.

## Authoring a `test_case.yaml`

```yaml
recording: recording.json          # path relative to this fixture folder
description: |
  Optional human-readable description of the scenario.
tags:                              # OPTIONAL — must be in pyproject.toml markers
  - simulated
  - kubernetes
expected_output:                   # OPTIONAL — defaults to ground_truth_diagnosis
  - "redis-master is OOMKilled"
  - "checkout-api is crashlooping"
include_tool_calls: false          # OPTIONAL — see /create-eval skill docs
skip: false                        # OPTIONAL
```

Optional advanced fields:

- `use_recording_system_prompt: bool` (default `true`) — if false, build the
  system prompt from HolmesGPT's normal `build_initial_ask_messages` flow
  using the loaded toolsets instead of the recording's prompt.
- `use_recording_user_prompt: bool` (default `true`) — if false, requires
  `user_prompt:` to be set on the test case.
- `toolsets: { name: { enabled: true, config: {...} } }` — same as `test_ask_holmes`.
- `toolsets_config_path` — path to a `toolsets.yaml` (matrix support).

## Running

```bash
# Smoke / unit tests (no LLM required)
poetry run pytest tests/llm/simulation -m "not llm" --no-cov

# One real simulated eval
SIMULATOR_MODEL="anthropic/claude-sonnet-4-5-20250929" \
poetry run pytest -k "01_checkout_redis_oom" --no-cov -vv

# All simulated evals
poetry run pytest tests/llm/test_simulated_env.py --no-cov

# Filter by tag
poetry run pytest -m "simulated" --no-cov
```

## Environment variables

| Var                    | Purpose                                                       |
|------------------------|---------------------------------------------------------------|
| `SIMULATOR_MODEL`      | LiteLLM model id for the simulator. Defaults to `MODEL`.      |
| `MODEL`                | Agent-under-test model (existing eval var).                   |
| `CLASSIFIER_MODEL`     | Judge model for `evaluate_correctness` (existing eval var).   |
| `BRAINTRUST_API_KEY`   | If set, eval results are logged to Braintrust (existing).     |

## Design choices

- **Recording schema is minimal** — `system_prompt + user_prompt + tool_calls + ground_truth_diagnosis`. Modeled on Simia-RL's `system + conversations` but specialized for HolmesGPT's single-question shape.
- **`SimulatedTool` wrapper class, not monkey-patching** — reversible by construction; the original toolset is never mutated, so other tests in the same pytest session can still use real tools.
- **Inner ReAct loop with meta-tools** (`fetch_recorded_response`, `search_recordings`) — the simulator-LLM pulls recorded responses on demand instead of having every recorded response stuffed into its system prompt. Necessary because real customer sessions can have multi-megabyte tool outputs.
- **Replay-only, no recorder in v1** — this module consumes recordings; capturing them from production is out of scope. If you need auto-capture, the easiest hook is `holmes/core/litellm_callbacks.py` (the same place Langfuse/Opik integrate).

## Caveats

See [`CAVEATS.md`](CAVEATS.md) for porting notes, drift risks, and operational
caveats. **Read it before applying this harness to a fork of HolmesGPT.**
