from ssa.scorers.numeracy import NumeracyScorer
from ssa.scorers.od_attr_bind import OdAttrBind
from ssa.scorers.spatial import SpatialScorer
from ssa.scoring import (
    NUMERACY,
    OD_ATTR_BINDING,
    SCORING_METHODS,
    SPATIAL,
)
from ssa.utils.logging import get_log

log = get_log(__file__)


class ModelScorer:
    def __init__(self, key, config_overrides=None, model=None):
        self.key = key
        log.info(f"Initializing Scorer {key}")
        if model:
            self.model = model
        elif key == SPATIAL:
            self.model = SpatialScorer(config_overrides=config_overrides)
        elif key == NUMERACY:
            self.model = NumeracyScorer(config_overrides=config_overrides)
        elif key == OD_ATTR_BINDING:
            self.model = OdAttrBind(config_overrides=config_overrides)
        else:
            raise Exception(
                f"Unrecognized scorer: {key}.  Choose from: {SCORING_METHODS}"
            )

    def warmup(self, wait=False):
        if hasattr(self.model, "warmup"):
            log.debug(f"Warm-up: {self.key}")
            self.model.warmup(wait=wait)
        else:
            log.debug(f"No warmup for {self.key}")

    def set_trace_collector(self, trace_collector):
        """Set the trace collector for the underlying scorer model."""
        if hasattr(self.model, "set_trace_collector"):
            self.model.set_trace_collector(trace_collector)

    def __str__(self):
        return f"Scorer:{self.model}"

    def __repr__(self):
        return f"Scorer:{type(self.model).__name__}"

    def config(self):
        if hasattr(self.model, "get_config_dict"):
            return self.model.get_config_dict()
        elif hasattr(self.model, "config") and hasattr(self.model.config, "dict"):
            return self.model.config.dict()
        elif hasattr(self.model, "config"):
            return getattr(self.model.config, "__dict__", {})
        return {}

    def evaluate(self, image, input_prompt, eval_criteria=None, **kwargs):

        # All scorers now use standardized BaseScorer interface
        return self.model.evaluate(
            image=image,
            input_prompt=input_prompt,
            eval_criteria=eval_criteria,
            **kwargs,
        )
