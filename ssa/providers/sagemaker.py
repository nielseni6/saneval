import base64
import io
import json
import re
import time
from typing import Dict

from botocore.exceptions import ClientError, ReadTimeoutError
from jsonschema import ValidationError as JsonValidationError
from PIL import Image
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ssa.aws import (
    setup_aws,
)
from ssa.aws.pricing import (
    AWS_ML_G4DN_XLARGE_COST,
    AWS_ML_G6E_XLARGE_COST,
    AWS_ML_M5_LARGE_COST,
)
from ssa.aws.sagemaker import LOCAL_DEBUG_URL, invoke_endpoint
from ssa.interfaces import BaseImageEditingMixin
from ssa.schemas import validate_response_against_schema
from ssa.utils.aspect_ratios import resolve_aspect_ratio_and_dimensions
from ssa.utils.costs import report_cost
from ssa.utils.http import RetryDecorator
from ssa.utils.images import resize_image
from ssa.utils.logging import get_log
from ssa.utils.system import apply_overrides

log = get_log(__file__)

LOCAL_IG = "local/ig"
LOCAL_VLM = "local/vlm"
AWS_MOCK_IG = "aws/mock-ig"
AWS_PIXART_SIGMA_900m = "aws/pixart-sigma-900m"
AWS_SANA_1_0 = "aws/sana-1-0"
AWS_SANA_1_5 = "aws/sana-1-5"
AWS_SANA_SPRINT = "aws/sana-sprint"
AWS_SSA = "aws/2ndsetai"
AWS_KANDINSKY_3_1 = "aws/kandinsky-3-1"
AWS_QWEN_2_5_7B = "aws/qwen-2-5-7b"
AWS_QWEN_2_5_7B_VLLM = "aws/qwen-2-5-7b-vllm"
AWS_PICK_SCORE = "aws/pick-score"
AWS_SANA_1600M_CNET = "aws/sana-1600m-cnet"
AWS_INTERNVL_25_8B = "aws/internvl-25-8b"


class SagemakerConfig(BaseModel):
    endpoint: str = None
    cost_per_img: float = None
    cost_per_hour: float = None


class SagemakerIgConfig(SagemakerConfig):
    pass


class SagemakerIgEditorConfig(SagemakerIgConfig):
    pass


class SagemakerVlmConfig(SagemakerConfig):
    resize_image: bool = True
    pass


class SagePixartConfig(SagemakerIgConfig):
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 3.0


class SageSanaConfig(SagemakerIgConfig):
    negative_prompt: str = ""
    num_inference_steps: int = 18
    guidance_scale: float = 5.0
    pag_guidance_scale: float = 2.0


class SageSana15Config(SagemakerIgConfig):
    num_inference_steps: int = 20
    guidance_scale: float = 4.5


class SageSanaCtrlnetConfig(SagemakerIgEditorConfig):
    guidance_scale: float = 4.5
    num_inference_steps: int = 10
    sketch_thickness: int = 2


class SageSanaSprintConfig(SagemakerIgConfig):
    num_inference_steps: int = 2
    guidance_scale: float = 4.5


class SageSsaConfig(SagemakerIgConfig):
    config_overrides: Dict[str, str] = None


class SageQwenConfig(SagemakerVlmConfig):
    batch_size: int = 5
    include_explanation: bool = False


class SageQwenVllmConfig(SagemakerVlmConfig):
    batch_size: int = 5
    include_explanation: bool = False
    max_tokens: int = 128
    temperature: float = 0.2
    system_prompt: str = "You are a helpful assistant."


class SageInternVLConfig(SagemakerVlmConfig):
    pass


class SagePickScoreConfig(SagemakerVlmConfig):
    max_length: int = 77


SAGEMAKER_IG_VERSIONS = {
    LOCAL_IG: SagemakerIgConfig(endpoint=LOCAL_DEBUG_URL, cost_per_hour=0.001),
    AWS_MOCK_IG: SagemakerIgConfig(
        endpoint="mock-ig-endpoint-001", cost_per_hour=AWS_ML_M5_LARGE_COST
    ),
    AWS_PIXART_SIGMA_900m: SagePixartConfig(
        endpoint="pixart-sig900-endpoint-013",
        cost_per_hour=AWS_ML_G6E_XLARGE_COST,
    ),
    AWS_SANA_1_0: SageSanaConfig(
        endpoint="sana-10-endpoint-018", cost_per_hour=AWS_ML_G6E_XLARGE_COST
    ),
    AWS_SANA_1_5: SageSana15Config(
        endpoint="sana-15-endpoint-001", cost_per_hour=AWS_ML_G6E_XLARGE_COST
    ),
    AWS_SANA_SPRINT: SageSanaSprintConfig(
        endpoint="sana-sprint-endpoint-001", cost_per_hour=AWS_ML_G6E_XLARGE_COST
    ),
    AWS_KANDINSKY_3_1: SagemakerIgConfig(
        endpoint="kandinsky-31-endpoint-007", cost_per_hour=AWS_ML_G6E_XLARGE_COST
    ),
    AWS_SSA: SageSsaConfig(
        endpoint="ssa-model-endpoint-006", cost_per_hour=AWS_ML_M5_LARGE_COST
    ),
    AWS_SANA_1600M_CNET: SageSanaCtrlnetConfig(
        endpoint="sana-1600m-cnet-endpoint-001", cost_per_hour=AWS_ML_G6E_XLARGE_COST
    ),
}

SAGEMAKER_VLM_VERSIONS = {
    LOCAL_VLM: SagemakerVlmConfig(endpoint=LOCAL_DEBUG_URL, cost_per_hour=0.001),
    AWS_QWEN_2_5_7B: SageQwenConfig(
        endpoint="qwen-25-7b-endpoint-005",
        cost_per_hour=AWS_ML_G6E_XLARGE_COST,
    ),
    AWS_QWEN_2_5_7B_VLLM: SageQwenVllmConfig(
        endpoint="qwen-25-7b-vllm-endpoint-010",
        cost_per_hour=AWS_ML_G6E_XLARGE_COST,
    ),
    AWS_INTERNVL_25_8B: SageInternVLConfig(
        endpoint="internvl-25-8b-endpoint-001",
        cost_per_hour=AWS_ML_G6E_XLARGE_COST,
    ),
}

# Things that dont serve as general purpose IGs/VLMs
SAGEMAKER_OTHER_VERSIONS = {
    AWS_PICK_SCORE: SagePickScoreConfig(
        endpoint="pick-score-endpoint-002",
        cost_per_hour=AWS_ML_G4DN_XLARGE_COST,
    )
}

SAGEMAKER_VERSIONS = {
    **SAGEMAKER_IG_VERSIONS,
    **SAGEMAKER_VLM_VERSIONS,
    **SAGEMAKER_OTHER_VERSIONS,
}


def fix_response(js):
    # Massage some quirks seen in LLM schematized response formatting
    if type(js) is not dict:
        return js

    if js and "properties" in js:
        js = js["properties"]
    if "required" in js:
        del js["required"]
    if "type" in js:
        del js["type"]

    for k, v in js.items():
        js[k] = fix_response(v)
    return js


def flex_load_schema(response_text: str):
    """
    Flexibly load and normalize a schema response from a model.

    Args:
        response_text (str): The text response from the model

    Returns:
        dict: The normalized JSON structure

    Raises:
        json.JSONDecodeError: If the response cannot be parsed as JSON
    """
    log.debug(f"Raw response: {response_text}")

    # Remove trailing commas that would cause JSON parsing to fail
    cleaned_text = re.sub(r",\s*([\]\}])", r"\1", response_text)

    # Sometimes formatted as ```json{ ... }```
    match = re.search(r"^```json\s*(.*?)```$", cleaned_text, re.DOTALL)
    if match:
        cleaned_text = match.group(1).strip()

    # Parse the JSON
    js = json.loads(cleaned_text)

    # Normalize the schema structure
    result = fix_response(js)

    # log.debug(f"Schema-loaded response: {result}")
    return result


class SagemakerProvider(BaseImageEditingMixin):
    def __init__(self, version=AWS_MOCK_IG, max_retries=8, config_overrides=None):
        assert version in SAGEMAKER_VERSIONS
        self.version = version
        self.max_retries = max_retries
        self.config = apply_overrides(
            SAGEMAKER_VERSIONS[self.version], config_overrides
        )
        self.endpoint = self.config.endpoint

        # Check/Config auth
        setup_aws()

    def __str__(self):
        return f"<SagemakerProvider:{self.version}>"

    def __repr__(self):
        return f"<SagemakerProvider:{self.version}>"

    def supports_image_editing(self) -> bool:
        """Most Sagemaker providers do not support image editing."""
        return False

    def report_call_cost(self, duration, metadata=None):
        # Report costs
        cost = 0.0
        metadata = metadata or {}
        if metadata_cost := metadata.get("cost"):
            # if metadata.cost is returned we assume this includes all costs
            # including the host/duration cost
            log.debug(f"Sagemaker returned cost in metadata: {metadata_cost}")
            cost += metadata_cost
        elif metadata_dur := metadata.get("gpu_duration"):
            log.debug(f"Sagemaker returned gpu_duration in metadata: {metadata_dur}")
            cost += metadata_dur * (self.config.cost_per_hour / 3600)
        else:
            if self.config.cost_per_img:
                cost += self.config.cost_per_img
            if self.config.cost_per_hour:
                cost += (self.config.cost_per_hour / 3600) * (duration)

        if cost:
            report_cost(cost, name=self.version)
        else:
            log.warning(f"Sagemaker {self} call had no defined costs {self.config}")

    def warmup(self, wait=False):
        try:
            invoke_endpoint(self.endpoint, {"warmup": True}, wait=wait)
        except Exception as e:
            log.warning(
                f"Model {self.version} did not handle warmup gracefully: {str(e)[0:90]}..."
            )

    @RetryDecorator(
        add_exceptions=(
            ClientError,
            ReadTimeoutError,
        )
    )
    def raw_call(self, request_data, image=None):
        request_data = request_data or {}

        # Encode image into data if present
        if image:
            buffer = io.BytesIO()
            image.save(buffer, format=image.format or "PNG")
            image_bytes = buffer.getvalue()
            base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
            request_data["image"] = base64_encoded

        # Add configs to call data
        for config_field, value in self.config:
            if config_field not in SagemakerConfig.model_fields.keys():
                request_data[config_field] = value

        t0 = time.time()
        response = invoke_endpoint(self.endpoint, request_data)
        t1 = time.time()

        # Decode response
        json_result = {}
        image = None
        metadata = None
        content_type = response.get("ContentType")
        if not content_type and type(response) is dict:
            json_result = response
            if img_base64 := json_result.get("image"):
                img_bytes = base64.b64decode(img_base64)
                image = Image.open(io.BytesIO(img_bytes))
                del json_result["image"]
            metadata = json_result.get("metadata", {})
        elif content_type == "image/png":
            image_bytes = response["Body"].read()
            image = Image.open(io.BytesIO(image_bytes))
            metadata = {}
        elif content_type == "application/json":
            response_body = response["Body"].read().decode("utf-8")
            json_result = json.loads(response_body)
            if img_base64 := json_result.get("image"):
                img_bytes = base64.b64decode(img_base64)
                image = Image.open(io.BytesIO(img_bytes))
                del json_result["image"]
            metadata = json_result.get("metadata", {})
        else:
            raise Exception(f"Unrecognized response {content_type}: {response}")

        # Report costs
        self.report_call_cost(t1 - t0, metadata)

        # add metadata to image
        if image:
            for k, v in metadata.items():
                image.info[k] = v

        return json_result, image

    def generate_image(
        self,
        prompt: str,
        width: int = 800,
        height: int = 800,
        seed: int = None,
        aspect_ratio: str = None,
        **kwargs,
    ) -> Image.Image:
        if any(kwargs.values()):
            raise NotImplementedError(f"Unsupported kwargs: {kwargs}")

        # Handle aspect_ratio parameter using unified resolution system
        if aspect_ratio is not None:
            # Resolve aspect_ratio to provider-specific dimensions and final ratio
            width, height, final_aspect_ratio = resolve_aspect_ratio_and_dimensions(
                aspect_ratio, self.version, width, height
            )

        if not isinstance(self.config, SagemakerIgConfig):
            raise Exception("Sagemaker provider {self} is not Image-generator")

        args = {
            "prompt": prompt,
            "image_size": {
                "width": width,
                "height": height,
            },
            "output_format": "png",
        }
        for config_field, value in self.config:
            if config_field not in SagemakerConfig.model_fields.keys():
                args[config_field] = value
        if seed:
            args["seed"] = seed

        _, image = self.raw_call(args, None)

        return image

    @RetryDecorator(
        exceptions=(
            PydanticValidationError,
            JsonValidationError,
            json.JSONDecodeError,
        )
    )
    def call(self, query, image=None, schema=None, seed=None, temperature=None):
        if not isinstance(self.config, SagemakerVlmConfig):
            raise Exception(f"Sagemaker provider {self} is not VLM")

        # Handle multiple images by taking the first one
        if image is not None and isinstance(image, list):
            if len(image) == 0:
                image = None
            else:
                if len(image) > 1:
                    log.warning(
                        f"Multiple images provided ({len(image)}), but sagemaker provider only supports single image. Using first image only."
                    )
                image = image[0]  # Take first image

        if image and self.config.resize_image:
            original_size = image.size
            if original_size[0] * original_size[1] > 1024 * 1024:
                log.debug("Rescaling image down")
                image = resize_image(image, (1024, 1024))
                new_size = image.size
                log.debug(f"Rescaled from {original_size} -> {new_size}")

        question = query
        if schema:
            # print(schema)
            if hasattr(schema, "model_json_schema"):
                schema_json = json.dumps(schema.model_json_schema())
            else:
                schema_json = json.dumps(schema)

            question = (
                f"{question}\n\nAnswer with a response in the schema: {schema_json}"
            )
            log.debug(f"Formatted sagemaker question: {question}")

        args = {}
        args["questions"] = [question]

        response, _ = self.raw_call(args, image=image)

        if "responses" in response:
            response = response["responses"][0]

        if schema:
            try:
                js = flex_load_schema(response)
            except json.JSONDecodeError:
                log.error(f"Bad response Json: {response}")
                raise

            validate_response_against_schema(js, schema)
            return js

        return response
