# Observability — Langfuse + Opik for HolmesGPT

Distilled from `~/gits/agent-scaffold/docs/observability_bundle/` (the bundle
was built for a LangChain agent; HolmesGPT is **not** LangChain — see "Why
not the bundle" below).

---

## TL;DR for HolmesGPT

HolmesGPT calls LiteLLM directly. LiteLLM has **first-party Langfuse + Opik
integrations** that intercept every `completion()` call:

```python
# Langfuse
import litellm
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
# Reads env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

# Opik
from opik.integrations.litellm import OpikLogger
litellm.callbacks = [OpikLogger(project_name="HolmesGPT-CLI")]
# Reads env vars: OPIK_API_KEY, OPIK_WORKSPACE, OPIK_PROJECT_NAME
```

That's the entire wiring. Both callbacks emit a complete generation +
tool-call trace per `completion()`, with token counts, latencies, model
name, request/response payloads. **No call-site changes needed in
HolmesGPT** — we just register the callbacks once at startup.

In this repo we wire those registrations through the existing
`Tracer.wrap_llm()` extension point in `holmes/core/tracing.py` so the
existing `--trace <provider>` CLI flag handles it. See
`holmesgpt-tracing-extension.md`.

---

## Why not the agent-scaffold bundle?

The bundle in `~/gits/agent-scaffold/docs/observability_bundle/` is built
around `langchain_core.callbacks.BaseCallbackHandler` — every provider
returns LangChain callbacks, threaded through `RunnableConfig`. Beautiful,
but it assumes LangChain. HolmesGPT does not use LangChain anywhere on the
hot path.

The LiteLLM-callback approach above is dramatically simpler for HolmesGPT
and gives you the same trace tree (generations + tool calls) in both
provider UIs.

---

## Required env vars

```bash
# Langfuse (cloud US default)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com   # NO trailing slash

# Opik (Comet cloud)
OPIK_API_KEY=...
OPIK_WORKSPACE=<your-workspace-slug>          # MUST be set non-interactively
OPIK_PROJECT_NAME=HolmesGPT-CLI               # whatever you want
```

We use the same credentials already configured for the agent-scaffold
project (US Langfuse + Comet workspace `gal-harari`). Stored in
`notes/.env.local` (gitignored).

---

## Gotchas (each cost an hour to discover in the original bundle)

### Langfuse

- **`LANGFUSE_HOST` must NOT have a trailing slash.** `…langfuse.com/`
  silently breaks the SDK.
- **v4 SDK API change:** uses `start_as_current_observation(as_type="span", …)`,
  not `start_as_current_span` from the v3 docs. (Doesn't affect us — the
  LiteLLM callback path bypasses this — but worth knowing if you ever
  hand-instrument spans.)
- **"0 tool calls" badge in trace list is a false negative** with some
  models. The data IS in the generation span's `output.tool_calls`; only
  the trace-list counter is wrong. Click into the generation.
- **Pin SDK to `^4.0.0`.** v5 may rename the LiteLLM callback string.

### Opik

- **`OPIK_WORKSPACE` MUST be set or `opik.configure()` hangs on stdin.**
  First call prompts: `Do you want to use "username" workspace? (Y/n)`.
  Non-interactive contexts (server, CI, agentic loop) will deadlock. Set
  the env var and we're fine.
- **Project name** is read from `OPIK_PROJECT_NAME` env var; you can also
  pass `project_name=` to `OpikLogger(project_name=...)` (we do).
- **Don't call `trace.end()` immediately after `trace()`** when batching
  is on (default) — risks data loss. The LiteLLM-callback path doesn't
  hit this; only matters for hand-rolled spans.
- **Pin SDK to `^1.5.0`.**

### LiteLLM (1.81.0 is what HolmesGPT pins)

- `litellm.success_callback` and `litellm.callbacks` are **separate lists**.
  Langfuse uses `success_callback` (string `"langfuse"`); Opik uses
  `callbacks` (an `OpikLogger` instance). Don't put one in the other's list.
- Adding the same callback twice will emit duplicate traces. Use
  `set(...)` or `isinstance` checks before appending (our tracer classes
  do this).
- `litellm.modify_params = True` (already set in HolmesGPT) lets the
  callbacks mutate the request — needed for some providers but harmless
  here.

### Network reachability

```bash
for h in us.cloud.langfuse.com www.comet.com; do
  curl -sf --max-time 5 -o /dev/null -w "%{http_code} %{remote_ip} $h\n" https://$h/
done
```

If either can't resolve, no amount of code-fiddling will help. Check VPN /
DNS first.

---

## Quick-verify after wiring

```bash
set -a; . ./notes/.env.local; set +a
poetry run python -c "
import os, litellm
litellm.success_callback = ['langfuse']
from opik.integrations.litellm import OpikLogger
litellm.callbacks = [OpikLogger(project_name=os.environ.get('OPIK_PROJECT_NAME','HolmesGPT-CLI'))]
r = litellm.completion(model='openai/qwen2.5:7b',
                       api_base='http://localhost:11434/v1',
                       api_key='dummy',
                       messages=[{'role':'user','content':'say hi'}])
print(r.choices[0].message.content)
"
# Then check both UIs for a fresh trace.
```

---

## Where things go in the UIs

- **Langfuse**: Project → Traces → sort by recent. Each `holmes ask` turn
  with multiple tool calls produces one trace with one root generation
  plus generations per follow-up turn.
- **Opik**: Workspace → Project → Traces. Same shape.

Both link the trace ID into the `litellm.completion()` response object so
you could in principle log the URL inline in the CLI; we don't bother —
the URL is the project page, and the trace is always the most recent.
