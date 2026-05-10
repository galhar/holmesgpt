# Simulated-env harness — caveats and porting notes

This document is for engineers (or agents) porting this harness to a fork of
HolmesGPT. The harness is intentionally small (~5 files), so most "drift" will
take the form of upstream HolmesGPT internals shifting under it. This file
enumerates each risk, how to detect it, and how to fix it.

## Drift risks

### #1 — `StructuredToolResult` field changes

**Where it bites you:** `RecordedToolCall.response` embeds
`holmes.core.tools.StructuredToolResult` verbatim. If upstream adds fields,
old recordings still validate (Pydantic falls back to defaults), but the
simulator may not populate the new fields and the agent under test could see
a partial result.

**How to detect:**

```bash
# Diff the StructuredToolResult definition between your fork and upstream
poetry run python -c "from holmes.core.tools import StructuredToolResult; print(list(StructuredToolResult.model_fields.keys()))"
```

**How to fix:** if a new field is *required* by the agent's downstream code,
extend the simulator's JSON output schema (in
`tests/llm/simulation/prompts/simulator_system.jinja2`) to mention it, and add
it to the parser in `simulator.py::_parse_final_answer`.

### #2 — `Tool.invoke` no longer calls `_apply_transformers` after `_invoke`

**Where it bites you:** the whole reason `SimulatedTool` overrides `_invoke`
(not `invoke`) is so the existing transformer chain — including the
`llm_summarize` transformer — runs on simulated output. If upstream moves the
transformer call elsewhere, the agent will see un-summarized simulated
responses, which differs from production behavior.

**How to detect:** run the bundled smoke test:

```bash
poetry run pytest tests/llm/simulation/tests/test_interceptor.py::test_transformer_pipeline_runs_on_simulated_output --no-cov -vv
```

The test registers a tiny `uppercase` transformer, wraps a tool, invokes it,
and asserts the simulated output is uppercased. If it fails, the transformer
chain is no longer firing.

**How to fix:** find where `_apply_transformers` is invoked now, and update
`SimulatedTool` to override the method that comes one level *above* the
transformer call (so the chain still runs after our delegate to the
simulator).

### #3 — Required field added to `Tool`

**Where it bites you:** `SimulatedTool.wrap()` in `interceptor.py` copies a
specific subset of fields (`name`, `description`, `parameters`,
`user_description`, `icon_url`, `transformers`, `restricted`). If upstream
adds a *required* field to `Tool`, `wrap` fails with a Pydantic
`ValidationError` at construction time.

**How to detect:** any simulated test fails immediately during setup with a
Pydantic error like `Field required [type=missing]`.

**How to fix:** add the new field to the `kwargs` dict in
`SimulatedTool.wrap()`. Do **not** copy private attributes (anything starting
with `_`) — those are recreated by `model_post_init`.

### #4 — `ToolExecutor.__init__` signature change

**Where it bites you:** `build_simulated_tool_executor` in `interceptor.py`
calls `ToolExecutor(new_toolsets)` with positional arg. If upstream adds a
new required arg (e.g. an event callback), this call breaks.

**How to detect:** any simulated test fails with `TypeError: __init__()
missing 1 required positional argument: 'foo'`.

**How to fix:** mirror `test_ask_holmes.py:190`, where the executor is
constructed for the live path. Whatever args it uses there, we need here too.

### #5 — A `Tool` subclass overrides `invoke()` itself (not just `_invoke`)

**Where it bites you (silent):** if upstream introduces a `Tool` subclass that
puts important logic in its overridden `invoke()` method, wrapping that tool
as a `SimulatedTool` will skip the override (because `SimulatedTool` inherits
the base `Tool.invoke`). The agent under test would behave differently than
in production, but no error is raised.

**How to detect:** run

```bash
grep -rn "def invoke(" /path/to/holmes/ | grep -v _invoke
```

If the only result is `holmes/core/tools.py::Tool.invoke`, you're safe. Any
additional match warrants inspection: does that override do something the
simulated path needs to preserve?

**How to fix:** if a subclass override is essential, replicate the relevant
behavior in `SimulatedTool` — usually by mirroring the override in
`SimulatedTool.invoke()` and then calling the base `Tool.invoke()` (which in
turn calls our `_invoke`).

### #6 — `Toolset.model_copy` drops private state

**Where it bites you:** Pydantic's `model_copy` in
`build_simulated_tool_executor` does a shallow copy. Pydantic private attrs
(`_initialized`, `_init_lock`, etc.) are NOT copied — they're re-created with
defaults. If a toolset relies on warm private state, the wrapped copy may
re-initialize.

**How to detect:** simulated tests behave correctly the first time, but break
when a toolset is re-used. Or: a `toolset.check_prerequisites` runs twice
(once for real, once for the wrapped copy).

**How to fix:** if needed, after `model_copy(update={"tools": ...})`,
explicitly copy the relevant private attrs from `ts` to `new_ts`. Currently
we already preserve `status` and `enabled` post-copy.

## Operational caveats

### Simulator model quality matters

The simulator is *another LLM*. With a weak `SIMULATOR_MODEL`, fabricated
responses can be:

- **Too consistent** — accidentally giving the agent the answer in the
  fabricated tool output. The agent "diagnoses" without earning it.
- **Inconsistent** — fabricated state contradicts what the simulator already
  said in a previous turn, making the scenario un-diagnosable.
- **Wrong-shape** — observed with `hermes3:8b` via the Ollama OpenAI shim:
  the simulator fetches a recorded response with `fetch_recorded_response`
  and forwards it *verbatim* even when the recorded tool's name doesn't
  match the tool the agent actually called (e.g. agent calls
  `fetch_resource_issues_metadata`, simulator returns kubectl_get_pods
  output). The simulator system prompt is explicit that this is forbidden
  (rule #1), but small models will still do it. Symptom in the trace: the
  agent's tool result `data` looks like an unrelated tool's output.

**Recommendation:** use Sonnet-4.5 / Opus-4.5 / GPT-5 class for the simulator.
Cheaper models (mini, haiku, 3.5-class) are usable for the smoke fixture but
unreliable for real evals.

### Cost

Each agent tool call triggers one simulator LLM call (plus 0–N meta-tool
calls when the simulator chooses to fetch recordings). A scenario that takes
8 tool calls in real mode costs roughly:

```
8 (agent) + 8 × ~3 (simulator + meta) = ~32 LLM calls
```

Budget accordingly. The agent's own loop dominates if the agent's model is
much more expensive than the simulator's.

### Determinism

The simulator runs at `temperature=0` by default for reproducibility. The
agent's LLM determinism is a separate concern (controlled by `MODEL` and the
agent's own LiteLLM args). For the most repeatable runs, use deterministic
models on both sides.

### Smoke test for the summary transformer

Keep one fixture (`_smoke_summary_transformer`) that:
- has a recorded response > 2000 chars,
- declares `transformers: [{name: llm_summarize, config: {input_threshold: 100}}]` in its `toolsets.yaml`,
- has an `expected_output` that includes a unique short token (e.g. a verification code) that survives summarization.

Run it after porting:

```bash
poetry run pytest -k "_smoke_summary_transformer" --no-cov -vv -s
```

Look in the logs for:

```
Applied transformer 'llm_summarize' to tool ...
```

If the line is absent, the summary transformer didn't fire — go back to
drift risk #2.

### Hand-authoring sanity check

After porting, change `ground_truth_diagnosis` in one fixture's
`recording.json` to nonsense and re-run:

```bash
poetry run pytest -k "01_checkout_redis_oom" --no-cov
```

The test should now FAIL. This proves that the gt is actually driving the
simulator's behavior (and not, e.g., that the agent is diagnosing from the
recorded responses leaking into context).

## Things this harness intentionally does NOT do

- Capture recordings from production. Author them by hand or build a thin
  recorder elsewhere. The starting hook is
  `holmes/core/litellm_callbacks.py`.
- Multi-turn conversation history. v1 supports a single `user_prompt` per
  recording. Add a `turns: list[Turn]` field to `Recording` if you need it.
- RL-style trajectory rewards (Simia-RL's specialty). The pass/fail signal
  here is the existing 0/1 `evaluate_correctness` judge — same as
  `test_ask_holmes`.
- Modify any code under `holmes/`. The entire harness lives under
  `tests/llm/simulation/` plus one new entry-point at
  `tests/llm/test_simulated_env.py` and one line in `pyproject.toml`'s
  pytest markers.

## File map

```
tests/llm/simulation/
├── README.md                     ← user-facing guide
├── CAVEATS.md                    ← THIS FILE — porting notes
├── __init__.py                   ← public re-exports
├── recording.py                  ← Pydantic schema + load/save
├── simulator.py                  ← LLM-as-simulator (inner ReAct loop)
├── interceptor.py                ← SimulatedTool wrapper + executor builder
├── prompts/
│   └── simulator_system.jinja2   ← simulator's system prompt template
└── tests/                        ← unit tests for the harness
    ├── test_recording.py
    ├── test_simulator.py
    └── test_interceptor.py

tests/llm/test_simulated_env.py   ← pytest entry-point (parallel to test_ask_holmes.py)
tests/llm/fixtures/test_simulated_env/
├── 01_checkout_redis_oom/
│   ├── test_case.yaml
│   └── recording.json
└── _smoke_summary_transformer/   ← drift-detection smoke fixture
    ├── test_case.yaml
    └── recording.json

pyproject.toml                    ← `simulated` pytest marker registered
```
