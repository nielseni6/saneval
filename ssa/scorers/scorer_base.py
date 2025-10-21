import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from PIL import Image
from pydantic import BaseModel

from ssa.utils.logging import get_log
from ssa.utils.system import apply_overrides

log = get_log(__file__)


class BaseScorerConfig(BaseModel):
    output_dir: Optional[str] = None  # Directory for saving debug outputs


class BaseScorer(ABC):
    def __init__(
        self,
        config: BaseScorerConfig = BaseScorerConfig(),
        config_overrides=None,
    ):
        self.config = apply_overrides(config, config_overrides)
        self.trace_collector = None

    def get_required_inputs(self) -> dict:
        """
        Returns a dictionary of required inputs for the create_eval_criteria method.
        """
        method = self.create_eval_criteria
        signature = inspect.signature(method)

        required_inputs = {}
        for param_name, param in signature.parameters.items():
            # A parameter is required if it has no default value.
            if param.default is inspect.Parameter.empty:
                required_inputs[param_name] = param.annotation

        return required_inputs

    @property
    def uses_eval_criteria(self) -> bool:
        """Override this to False for scorers that don't use evaluation criteria (e.g., PickScore, HpsV2)."""
        return True

    @abstractmethod
    def create_eval_criteria(self, *args, **kwargs) -> Optional[Union[dict, List[str]]]:
        """Create a list of eval criteria for a given text(s) and image(s) input.

        Returns a dictionary (preferred) or a list of evaluation criteria for this scorer to use in evaluate.
        """

    @abstractmethod
    def evaluate(
        self,
        image: Optional[Union[Image.Image, List[Image.Image]]],
        input_prompt: Optional[Union[str, List[str]]],
        eval_criteria: Optional[Union[dict, List[str]]],
        context: Optional[dict] = None,
    ) -> Tuple[bool, float, List[str]]:
        """Evaluate the image(s) on the prompt(s) with the given eval_criteria, and eval_context.

        Args:
            image: A PIL image, or list of PIL images to be evaluated
            input_prompt: The text prompt, or list of text prompts to evaluate against the image(s)
            eval_criteria: Optional evaluation criteria to use.

        Returns:
            Tuple containing:
                - correct: Boolean indicating if all criteria passed or the score is higher than the threshold
                - score: Float score for the evaluation
                - non_conforming: List of unsatisfied/non-conforming criteria
        """

    @property
    @abstractmethod
    def scorer_name(self) -> str:
        """Return the name of this scorer."""
        pass

    @property
    def supported_metrics(self) -> List[str]:
        """Return list of supported metrics. Subclasses should define a `_metrics` list attribute."""
        return getattr(self, "_metrics", [])

    @abstractmethod
    def warmup(self, wait=False) -> None:
        """Warmup any underlying models."""

    def __str__(self):
        return f"Scorer:{self.key}"

    def __repr__(self):
        return f"Scorer:{type(self).__name__}"

    @property
    def key(self) -> str:
        """Return a key identifier for this scorer. Defaults to scorer_name."""
        return self.scorer_name

    def get_config_dict(self) -> dict:
        """Return the configuration as a dictionary."""
        if hasattr(self.config, "model_dump"):
            return self.config.model_dump()
        elif hasattr(self.config, "dict"):
            return self.config.dict()
        elif hasattr(self.config, "__dict__"):
            return self.config.__dict__
        else:
            return {}

    def set_trace_collector(self, trace_collector):
        """Set the trace collector for the scorer and its sub-models."""
        self.trace_collector = trace_collector
        # Propagate to common sub-models if they exist and support it
        for attr_name in ["llm", "vlm", "object_detector"]:
            if hasattr(self, attr_name):
                model = getattr(self, attr_name)
                if hasattr(model, "set_trace_collector"):
                    model.set_trace_collector(trace_collector)
                elif hasattr(model, "trace_collector"):
                    # Fallback for models that have the attribute but not the method
                    model.trace_collector = trace_collector
