"""
Simple reasoning trace collection for LLM/VLM calls during benchmarking.
"""

from typing import Any, Dict, List, Optional

from ssa.utils.logging import get_log

log = get_log(__file__)


def default_json_handler(obj):
    """
    Default JSON handler for non-serializable objects in reasoning traces.

    This function handles common non-serializable objects that might appear
    in LLM/VLM responses, such as Pydantic models, custom classes, etc.
    """
    # Handle Pydantic models
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    # Handle other objects with __dict__
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    # Fallback to string representation
    else:
        return str(obj)


class ReasoningTraceCollector:
    """Collects reasoning traces for each prompt during benchmark runs."""

    def __init__(self):
        self.traces: List[Dict[str, Any]] = []
        self._current_prompt_trace: Optional[Dict[str, Any]] = None

    def start_prompt(self, prompt_id: str, prompt_text: str) -> None:
        """Start collecting traces for a new prompt."""
        self._current_prompt_trace = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "calls": [],
        }

    def record_call(
        self,
        model_type: str,  # "llm" or "vlm"
        model_key: str,
        query: str,
        response: Any = None,
        scorer_name: Optional[str] = None,
        step_description: Optional[str] = None,
    ) -> None:
        """Record a single LLM/VLM call."""
        if not self._current_prompt_trace:
            log.warning("record_call invoked without an active prompt trace. Ignoring.")
            return

        call_data = {
            "model_type": model_type,
            "model_key": model_key,
            "scorer_name": scorer_name,
            "step_description": step_description,
            "query": query,
            "response": response,
        }

        self._current_prompt_trace["calls"].append(call_data)

    def finish_prompt(self) -> None:
        """Finish collecting traces for current prompt and add to final list."""
        if self._current_prompt_trace:
            self.traces.append(self._current_prompt_trace)
            self._current_prompt_trace = None

    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all collected traces."""
        return self.traces

    def clear(self) -> None:
        """Clear all traces."""
        self.traces = []
        self._current_prompt_trace = None
