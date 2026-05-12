"""LLM-as-simulator harness for HolmesGPT eval tests.

See ``tests/llm/simulation/README.md`` for a full overview. In short:

- ``recording.py`` defines the JSON scenario format (system prompt, user prompt,
  recorded tool calls with real responses, ground-truth diagnosis).
- ``simulator.py`` is the LLM-as-simulator: given an agent tool call, it returns
  a fabricated ``StructuredToolResult`` consistent with the recording.
- ``interceptor.py`` wraps each ``Tool`` in a ``SimulatedTool`` whose ``_invoke``
  delegates to the simulator. The original ``Tool.invoke`` pipeline (parameter
  coercion, approval, transformer chain incl. ``llm_summarize``) is preserved.

The harness lives entirely under ``tests/llm/`` and does not modify ``holmes/``.
"""

from tests.llm.simulation.interceptor import (
    SimulatedTool,
    build_simulated_tool_executor,
)
from tests.llm.simulation.recording import (
    RecordedToolCall,
    Recording,
    load_recording,
    save_recording,
)
from tests.llm.simulation.simulator import ToolCallSimulator

__all__ = [
    "RecordedToolCall",
    "Recording",
    "SimulatedTool",
    "ToolCallSimulator",
    "build_simulated_tool_executor",
    "load_recording",
    "save_recording",
]
