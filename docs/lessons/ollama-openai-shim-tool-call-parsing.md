# Ollama's OpenAI-compatible shim drops qwen tool calls — use `ollama_chat/` instead

A grounded debugging note from running the HolmesGPT eval suite against
qwen2.5:7b served by Ollama. Captures what the model emits, where the
tool-call format is defined, and which translation layer is dropping it.

## Symptom

Eval runs against `MODEL=openai/qwen2.5:7b` (LiteLLM `openai` provider hitting
Ollama's `/v1/chat/completions` shim) showed assistant turns with **empty
content AND no tool calls**:

```python
output = {
    "content": "",
    "role": "assistant",
    "tool_calls": None,
    "function_call": None,
    "provider_specific_fields": {"refusal": None},
}
# usage: prompt=14684, completion=51, total=14735
# finish_reason="stop"
```

`completion=51` confirms the model emitted 51 tokens. Those tokens never
reach the caller. Tests fail because Holmes's agent loop can't continue
without either a tool call or a final answer.

## Where qwen2.5's tool-call format is defined

It's the model's own chat template, baked into the GGUF / `Modelfile`
that Ollama loads. Exact field: `chat_template` in
[Qwen2.5-7B-Instruct's `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)
on Hugging Face. Ollama doesn't invent a template — it uses qwen's.

When `tools` is non-empty, the template injects this system block:

```
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"function": {...}, "type": "function"}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

So when qwen wants to call a tool, it emits exactly:

```
<tool_call>
{"name": "kubernetes_jq_query", "arguments": {"kind": "pods", "jq_expr": "..."}}
</tool_call>
```

Literal `<tool_call>` XML tags wrapping a JSON object. **Not** OpenAI's
structured `tool_calls` field — qwen produces text. Some downstream layer
has to extract it.

## Where the bug lives — three layers, two translation boundaries

Both endpoints route to the **same** Ollama generation backend with the
**same** chat template. The difference is at the boundaries.

| Layer | What it does | `openai/` path | `ollama_chat/` path |
|---|---|---|---|
| Request translation: OpenAI msg → template variables | Parses `tool_calls.function.arguments` (OpenAI sends a JSON-string, qwen template wants a dict to `tojson`) | Ollama's Go shim, `openai/openai.go` | LiteLLM's Python, `litellm/llms/ollama/chat/transformation.py:266-282` |
| Chat template rendering | Renders qwen's Jinja template into the literal token stream | Ollama Go (same code path) | Ollama Go (same code path) |
| Response translation: model output → OpenAI msg | Recognizes `<tool_call>...</tool_call>` blocks, extracts JSON, packs into `tool_calls` array, sets `finish_reason="tool_calls"` | Ollama's Go shim | LiteLLM's Python, `litellm/llms/ollama/chat/transformation.py:380-405` |

The middle layer (template rendering) is identical. The bug is at one or
both translation boundaries — specifically inside Ollama's openai shim.

## Concrete demonstration

Same input messages + 22 tools, same Ollama process, same model file:

```python
# Path 1: openai/ via Ollama's openai-compat shim
client = OpenAI(api_key='dummy', base_url='http://localhost:11434/v1')
r = client.chat.completions.create(model='qwen2.5:7b', messages=msgs, tools=tools)
# → finish_reason='stop', content='', tool_calls=None, completion_tokens=51  ❌

# Path 2: ollama_chat/ via LiteLLM's native Ollama provider
import litellm
r = litellm.completion(
    model='ollama_chat/qwen2.5:7b',
    api_base='http://localhost:11434',
    messages=msgs, tools=tools,
)
# → finish_reason='tool_calls', content='', tool_calls=[ChatCompletionMessageToolCall(...)]  ✅
```

The 51 vs 133 completion-token difference is the chat-template overhead
(slightly different framing); the model is doing the same work. The
shim's parser eats the result, LiteLLM's parser doesn't.

## Reading list (where to look next time)

| File | Lines | What it shows |
|---|---|---|
| `litellm/llms/ollama/chat/transformation.py` | 219-242 | `get_complete_url` — `ollama_chat/` hits `{api_base}/api/chat`, NOT `/v1/...` |
| `litellm/llms/ollama/chat/transformation.py` | 244-322 | `transform_request` — converts OpenAI tool_calls to Ollama's `OllamaToolCall` (`arguments` STRING → DICT) |
| `litellm/llms/ollama/chat/transformation.py` | 380-410 | `transform_response` — recognises Ollama's structured tool-call output and rebuilds OpenAI shape, sets `finish_reason="tool_calls"` |
| `litellm/main.py` | 3792-3829 | `ollama_chat` provider dispatch — env var resolution: `OLLAMA_API_BASE` → fallback `http://localhost:11434` |
| `litellm/main.py` | 1947-1963 | `openai` provider dispatch — env vars: `OPENAI_BASE_URL` → `OPENAI_API_BASE` → `https://api.openai.com/v1` |
| `litellm/litellm_core_utils/get_llm_provider_logic.py` | 138-180 | `get_llm_provider` — splits model on `/` to pick provider |
| `tokenizer_config.json` (HF, Qwen2.5-7B-Instruct) | `chat_template` | The Jinja that defines the `<tool_call>...</tool_call>` protocol |

## Action

For HolmesGPT against an Ollama-served model, set:

```bash
export MODEL=ollama_chat/qwen2.5:7b
export OLLAMA_API_BASE=http://localhost:11434
# CLASSIFIER_MODEL stays as the raw model name (no provider prefix) — the
# eval classifier uses the OpenAI SDK directly, not litellm, so it sends
# the model name verbatim to Ollama's /v1/chat/completions.
export CLASSIFIER_MODEL=qwen2.5:7b
```

Do **not** use `MODEL=openai/qwen2.5:7b` with `OPENAI_API_BASE=…/v1` — that
path goes through Ollama's openai shim and silently drops tool calls.

## Open questions

- The shim's exact failure mode (which inputs trigger empty output vs which
  produce a usable tool call) isn't fully characterised here. The reproduction
  above used a 4-message conversation with one prior assistant tool call and
  one tool response. Simpler 2-message inputs sometimes work via the shim.
  Suspect: nested-JSON `arguments` re-encoding loses structure when the shim
  re-emits the chat-template variables.
- Likely related upstream issues:
  - <https://github.com/ollama/ollama/issues/4555>
  - <https://github.com/BerriAI/litellm> Ollama-related issues filed against
    litellm 1.81.x
- This may already be fixed in a newer Ollama; the GGUF runner's
  tool-call parsing has been actively churning. Worth retesting after an
  `ollama pull` of a fresh server build.
