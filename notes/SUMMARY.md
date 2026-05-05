# Lessons & problems — HolmesGPT × Newton × Langfuse + Opik

Last updated: 2026-05-05 11:50 (C6 commit; LLM full-run continues in background)

## Lessons

- HolmesGPT calls **LiteLLM directly** (no LangChain) → wire Langfuse/Opik
  via LiteLLM's first-party callbacks/decorators, not via LangChain handlers.
  Zero call-site changes needed in `tool_calling_llm.py`.
- HolmesGPT's `holmes/core/tracing.py::TracingFactory.create_tracer` is the
  one extension point — every new provider drops in there.
- `qwen2.5:7b` is the right local default (~3 s warm, decent tool calls).
  Newton PRO6000 partition is usually full → use `--gres=gpu:L40:1` (lots of
  L40 capacity); `qwen3.5:35b` is 10–30× slower and impractical for tests.
- HolmesGPT eval tests' `before_test` hooks `kubectl apply` real manifests;
  ~157 of 248 LLM eval cases need a local k8s cluster.

## Problems encountered & fixed

- **Newton PRO6000 partition fully booked** (PD/Resources for 5 min) →
  **fix:** override `--gres=gpu:L40:1` at sbatch submit (L40 has 30+ free
  GPUs across the cluster). Job came up in ~30 s.
- **Newton sbatch script's READY marker is stale** — script prints
  `=== SETUP COMPLETE ===` not `=== READY ===` (doc was older). Also the
  log message claims it exposes login:11434 but the actual `-R` flag
  tunnels to login:**11435**. → **fix:** read the real script with
  `grep -nE '\-R|tunnel' ~/ollama_server.sh` to get the truth.
- **`langfuse v4` breaks `litellm 1.81.0`'s built-in callback** —
  `module 'langfuse' has no attribute 'version'`. v4 SDK dropped the
  `langfuse.version` submodule that litellm 1.81.0 calls into. → **fix:**
  pin `langfuse = "^2.60.0"`. Bump together with a future litellm bump.
- **`opik 1.10+` removed `OpikLogger` callback class** — `cannot import
  name 'OpikLogger'`. Current API is `track_completion(project_name=...)`
  decorator. → **fix:** rewrote `OpikTracer.wrap_llm` to monkey-patch
  `litellm.completion` with the decorator; idempotent via `_opik_tracked`
  attribute.
- **Poetry not installed on this laptop** — `pip install poetry` first.
  Created venv at `/home/gal/.cache/pypoetry/virtualenvs/holmesgpt-TewFsB5r-py3.13`.
- **HolmesGPT eval `shared_test_infrastructure` is session-scoped + autouse**
  (`tests/llm/conftest.py:130`) — at session start it runs `before_test` for
  ALL collected tests in parallel via `run_all_test_setup()`, regardless of
  `-n`. With 183 tests this created ~106 namespaces and 2200+ pods in
  CrashLoop/Init state on a 3-node kind cluster. → **fix:** set
  `RUN_LIVE=false` to skip live setup entirely (line 138 guard); tests use
  mocks for tools but still call real Newton LLM. Trade-off: doesn't
  validate the kubectl integration, but validates Holmes's reasoning.
- **`pytest --strict-setup-mode` flag bug** — pytest treated it as needing
  an arg. conftest accepts `--strict-setup-mode=true|false` (=value form).
  → just dropped the flag; default is false.
- **CLAUDE.md says "ALWAYS create commits, NEVER amend"** → the langfuse v2
  pin is a separate commit (C2b) on top of C2 instead of an amend. Same for
  the Opik fix (C4b on top of C4). User can cherry-pick (C2 + C2b) or
  (C4 + C4b) together.

## Test results

### Non-LLM suite (`make test-without-llm`)
- **2149 passed / 6 failed / 81 skipped** in 2:21 (log: `test-runs/test-without-llm-20260505-102331.log`).
- The 6 failures are all infrastructure-dependent and unrelated to our tracing
  changes:
  - 4 × Grafana toolset health-check tests assume `localhost:3000` is Grafana,
    but on this machine port 3000 is held by the user's `agent-scaffold-openwebui`
    container. (Tests passed pre-existing `check_service_running("Grafana", 3000)`
    TCP probe but then got HTML instead of JSON when calling Grafana APIs.)
  - 2 × Tempo toolset health-check tests need a Tempo deployment on
    `localhost:3200`. The local kind cluster has Prometheus + Loki + Grafana
    but not Tempo. After port-forwarding Loki to localhost:3100, **5 failed /
    5 passed** in the rerun (`test-grafana-rerun-20260505-102750.log`).

### LLM eval suite (`tests/llm/test_ask_holmes.py`)
- 4 progressive attempts, each fixing a different blocker:
  1. `--strict-setup-mode` (no value) → pytest reject. Dropped flag.
  2. All 183 SKIPPED — classifier hits Ollama with `openai/qwen2.5:7b`
     model name verbatim; Ollama only knows `qwen2.5:7b`.
     → set `CLASSIFIER_MODEL=qwen2.5:7b` (raw name, no prefix).
  3. With `RUN_LIVE=true` (default): conftest's session-scoped
     `shared_test_infrastructure` ran `before_test` for ALL 183 tests
     upfront, ignoring `-n` parallelism → 106 namespaces and 2278 pods
     in non-Running state on the 3-node kind cluster after 5 min. Killed.
  4. With `RUN_LIVE=false`: tests use mocks for tools but still call
     real Newton LLM. Avoids cluster overload. Each test takes ~3 min
     against qwen2.5:7b (Holmes call + classifier grading round).
     183 × 3 min ÷ 4 workers ≈ 137 min. Killed after 5 min (1 fail).
- Pragmatic narrowing: ran a 5-test subset serially (`01_…05_`) for a
  meaningful logged result. See `test-runs/test-llm-ask-holmes-subset-*.log`.
  **5/5 failed in 62s** — all rubric-fail.
- Final full run with `-n 2`, `RUN_LIVE=false`:
  `test-runs/test-llm-ask-holmes-full-20260505-114159.log`. At C6 commit
  time, 12/183 done (all FAILED). Continuing in the background; the
  final summary will append to that log when pytest exits.
- **Combined LLM result (so far):** 17 tests run, **0 PASSED / 17 FAILED**
  — consistent with the documented "HolmesGPT with local 7B → ~0% pass
  rate, works only with cloud LLMs" finding from
  `~/gits/k8s-ai-agent-benchmark/docs/local-setup.md`. The framework
  works end-to-end; the model is the bottleneck.

## Open / unresolved

- 4 Grafana tests + 2 Tempo tests can't be made to pass without (a) freeing
  port 3000 on the host (would require stopping the user's OpenWebUI
  container), (b) port-forwarding Grafana to 3000, and (c) deploying Tempo
  in the kind cluster. Skipped for this run.
