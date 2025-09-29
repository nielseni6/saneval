from ssa.scorers.bag_of_criteria_vlm import BagOfCriteriaVlm as BocVlm
from ssa.scorers.dsg_vlm import DsgVlm
from ssa.scorers.face_analysis_scorer import FaceAnalysisScorer
from ssa.scorers.face_id import FaceId
from ssa.scorers.hpsv2 import HpsV2
from ssa.scorers.i2i_criteria_vlm import I2ICriteriaVlm
from ssa.scorers.mock_i2i import MockI2I
from ssa.scorers.numeracy import NumeracyScorer
from ssa.scorers.od_attr_bind import OdAttrBind
from ssa.scorers.pick_score import PickScore
from ssa.scorers.roundtrip import Roundtrip
from ssa.scorers.spatial import SpatialScorer
from ssa.scoring import (
    BOC_VLM,
    COMPBENCH,
    DSG_VLM,
    FACE_ANALYSIS,
    FACE_ID,
    GEN_EVAL,
    HPSV2,
    I2I_CRITERIA_VLM,
    MOCK_I2I,
    NUMERACY,
    OD_ATTR_BINDING,
    PICK_SCORE,
    ROUNDTRIP,
    SCORING_METHODS,
    SPATIAL,
)
from ssa.thirdparty.compbench.spatial_numeracy_eval import CompBench
from ssa.thirdparty.geneval.evaluate_images import GenEval
from ssa.utils.logging import get_log

log = get_log(__file__)


class ModelScorer:
    def __init__(self, key, config_overrides=None, model=None):
        self.key = key
        log.info(f"Initializing Scorer {key}")
        if model:
            self.model = model
        elif key == DSG_VLM:
            # DSG + VLM based criteria generation/eval
            self.model = DsgVlm(config_overrides=config_overrides)
        elif key == BOC_VLM:
            # BOC (bag of criteria / LLM) /VLM based criteria generation/eval
            self.model = BocVlm(config_overrides=config_overrides)
        elif key == GEN_EVAL:
            self.model = GenEval(config_overrides=config_overrides)
        elif key == HPSV2:
            self.model = HpsV2(config_overrides=config_overrides)
        elif key == PICK_SCORE:
            self.model = PickScore(config_overrides=config_overrides)
        elif key == COMPBENCH:
            self.model = CompBench(config_overrides=config_overrides)
        elif key == SPATIAL:
            self.model = SpatialScorer(config_overrides=config_overrides)
        elif key == NUMERACY:
            self.model = NumeracyScorer(config_overrides=config_overrides)
        elif key == OD_ATTR_BINDING:
            self.model = OdAttrBind(config_overrides=config_overrides)
        elif key == FACE_ANALYSIS:
            self.model = FaceAnalysisScorer(config_overrides=config_overrides)
        elif key == FACE_ID:
            self.model = FaceId(config_overrides=config_overrides)
        elif key == ROUNDTRIP:
            self.model = Roundtrip(config_overrides=config_overrides)
        elif key == MOCK_I2I:
            self.model = MockI2I(config=config_overrides)
        elif key == I2I_CRITERIA_VLM:
            self.model = I2ICriteriaVlm(config_overrides=config_overrides)
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
