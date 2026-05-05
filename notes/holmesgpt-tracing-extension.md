# HolmesGPT tracing extension points

A reference for extending HolmesGPT with new tracing providers (Langfuse,
Opik, etc.). Captured so future-me (or another Claude) can wire a new
backend in 5 minutes without re-reading the codebase.

---

## The architecture in two paragraphs

`holmes/core/tracing.py` defines a tiny `Tracer` interface implemented by
`DummyTracer`, `BraintrustTracer`, `OpenTelemetryTracer`, and (now)
`LangfuseTracer` + `OpikTracer`. The interface is essentially just
`start_experiment / start_trace / get_trace_url / wrap_llm` — no formal
ABC, just duck-typed.

`TracingFactory.create_tracer(trace_type: str, project: str)` is a single
dispatch that the CLI calls once at startup. The CLI flag is
`--trace <provider>` (`holmes/main.py:232`) and the value gets passed in
verbatim. Inside the agent loop, `tool_calling_llm.py` calls
`self.tracer.wrap_llm(litellm)` once — that wrapped module is what every
`completion()` is dispatched through.

So a new tracer needs to either (a) wrap the litellm module to inject
behaviour, OR (b) register a global LiteLLM callback. Modern providers
(Langfuse, Opik) ship LiteLLM-native callbacks → option (b) is one line.

---

## File map

| Path | What's there |
|---|---|
| `holmes/core/tracing.py:107–151` | `DummySpan` — no-op span; new spans should expose the same surface (`start_span`, `log`, `end`, `set_attributes`, context-manager methods) |
| `holmes/core/tracing.py:153–171` | `DummyTracer` — minimal interface to copy |
| `holmes/core/tracing.py:173–299` | `BraintrustTracer` — example of the **wrap-the-module** pattern (`WrappedLiteLLM` at line 285) |
| `holmes/core/otel_tracing.py` | `OpenTelemetryTracer` — example of the **register-global-callback** pattern (uses OTel context propagation) |
| `holmes/core/tracing.py:302–380` | `TracingFactory.create_tracer` — add a new `if trace_type.lower() == "yourname":` branch here |
| `holmes/main.py:232–236` | `--trace` CLI option; just update the help text when adding providers |
| `holmes/main.py:302` | `tracer = TracingFactory.create_tracer(trace, project="HolmesGPT-CLI")` — single call site for the CLI |
| `holmes/core/tool_calling_llm.py:198–220` | `tracer` is a constructor parameter; passed down through `ToolCallingLLM` and `IssueInvestigator` |
| `holmes/core/tool_calling_llm.py:572`-ish | The actual `wrap_llm(litellm)` call site (find via `grep -n wrap_llm holmes/core/tool_calling_llm.py`) |

---

## Adding a new provider — 5-minute recipe

### Pattern A: provider has a LiteLLM-native callback (Langfuse, Opik, Helicone, etc.)

```python
class YourTracer:
    def __init__(self, project: str):
        self.project = project

    def start_experiment(self, experiment_name=None, additional_metadata=None):
        return None  # most LLMOps platforms organise by trace, not experiment-by-name

    def start_trace(self, name, span_type=None):
        return DummySpan()  # callback path creates the trace per-completion

    def get_trace_url(self):
        return "https://your-provider/project-page"  # link the user can click

    def wrap_llm(self, llm_module):
        import litellm
        # Pick the right list — provider-dependent
        litellm.success_callback = list(set(
            (litellm.success_callback or []) + ["yourname"]
        ))
        return llm_module
```

Then in `TracingFactory.create_tracer`:

```python
if trace_type.lower() == "yourname":
    if not os.environ.get("YOURNAME_API_KEY"):
        logging.warning("YourName tracing requested but YOURNAME_API_KEY not set")
        return DummyTracer()
    return YourTracer(project=project)
```

### Pattern B: provider needs a wrapped module (Braintrust)

See `BraintrustTracer.wrap_llm` (line 278). You replace `litellm.completion`
with a wrapper that augments inputs/outputs before calling through. More
intrusive but lets you add per-call metadata that LiteLLM callbacks can't see.

### Composite (multiple providers at once)

`CompositeTracer` (added by us in commit C5) accepts a list of tracers and
fans out every method to all of them. Triggered by passing a comma-separated
string to `--trace`, e.g. `--trace langfuse,opik`. Implementation in
`tracing.py`:

```python
class CompositeTracer:
    def __init__(self, tracers): self._tracers = tracers
    def start_experiment(self, *a, **kw):
        for t in self._tracers: t.start_experiment(*a, **kw)
    def start_trace(self, *a, **kw):
        for t in self._tracers:
            s = t.start_trace(*a, **kw)
            if not isinstance(s, DummySpan): return s
        return DummySpan()
    def get_trace_url(self):
        urls = [t.get_trace_url() for t in self._tracers]
        return " | ".join(filter(None, urls))
    def wrap_llm(self, llm_module):
        for t in self._tracers: llm_module = t.wrap_llm(llm_module)
        return llm_module
```

The dispatch then looks like:

```python
if trace_type and "," in trace_type:
    return CompositeTracer([
        TracingFactory.create_tracer(t.strip(), project)
        for t in trace_type.split(",")
    ])
```

---

## What does NOT need changing

- `tool_calling_llm.py` — never. The `wrap_llm(litellm)` call site is
  framework-agnostic; new providers slot in without touching it.
- `llm.py` (`DefaultLLM`) — never. It calls `litellm.completion()` whose
  callbacks fire automatically.
- The toolset machinery — never. Tools execute through HolmesGPT's own
  loop, not through the LLM provider, so tool-call traces come from
  LiteLLM's metadata about the function-calling response.

---

## Verification snippet

```bash
# After your new tracer is added:
poetry run python -c "
from holmes.core.tracing import TracingFactory
t = TracingFactory.create_tracer('yourname', project='HolmesGPT-CLI')
print(type(t).__name__, '→', t.get_trace_url())
"
# Should print 'YourTracer → https://...'
```

Then run a real `holmes ask` with `--trace yourname` and verify a trace
shows in your provider's UI.
