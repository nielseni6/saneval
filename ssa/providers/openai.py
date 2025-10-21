import base64
import json
from io import BytesIO

import openai
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

from ssa.interfaces import BaseImageEditingMixin
from ssa.utils.aspect_ratios import resolve_aspect_ratio_and_dimensions
from ssa.utils.images import (
    encode_image,
    save_image_to_disk,
)
from ssa.utils.logging import get_log
from ssa.utils.secrets import get_secret
from ssa.utils.system import apply_overrides

log = get_log(__file__)

# OpenAI supported image sizes (based on current API)
OPENAI_SUPPORTED_SIZES = [
    "1024x1024",  # 1:1 (ratio 1.0)
    "1536x1024",  # 3:2 (ratio 1.5)
    "1024x1536",  # 2:3 (ratio 0.667)
]

# OpenAI supported aspect ratios for easy reference
OPENAI_SUPPORTED_ASPECT_RATIOS = [
    "1:1",  # Square (1024x1024)
    "3:2",  # Landscape (1536x1024)
    "2:3",  # Portrait (1024x1536)
]

# Direct mapping from aspect ratios to OpenAI size strings
OPENAI_ASPECT_RATIO_TO_SIZE = {
    "1:1": "1024x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
}


GPT_4O = "openai/gpt-4o"
GPT_4_5 = "openai/gpt-4.5-preview"
O1 = "openai/o1"


class OpenAIConfig(BaseModel):
    key: str = ""
    max_retries: int = 7
    temperature: float = 0
    supported_sizes: list[str] = OPENAI_SUPPORTED_SIZES
    supported_aspect_ratios: list[str] = OPENAI_SUPPORTED_ASPECT_RATIOS


class OpenAITi2iConfig(OpenAIConfig):
    image: str = None


class OpenAIO1Config(BaseModel):
    key: str = ""
    max_retries: int = 7


# https://openai.com/api/pricing/
OPENAI_SUPPORTED_VERSIONS = {
    GPT_4O: OpenAIConfig(key="gpt-4o"),
    GPT_4_5: OpenAIConfig(key="gpt-4.5-preview"),
    O1: OpenAIO1Config(key="o1"),
}

OPENAI_IG_VERSIONS = {
    GPT_4O: OpenAIConfig(key="gpt-4o"),
}

OPENAI_I2I_SUPPORTED_MODELS = [
    GPT_4O,
]


def fix_json_schema(schema):
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        for k, v in schema.get("properties").items():
            fix_json_schema(v)


class OpenAIProvider(BaseImageEditingMixin):
    def __init__(
        self, version=GPT_4O, max_retries=5, config_overrides=None, api_key=None
    ):
        self.version = version
        self._api_key = api_key  # Store API key for potential direct use or mocking
        self.config = self._configure_provider(config_overrides)
        self.client = self._initialize_client()
        self.max_retries = max_retries  # Note: This overrides config.max_retries if provider is called with max_retries

    def _configure_provider(self, config_overrides):
        """Helper to apply configuration overrides."""
        if self.version not in OPENAI_SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported OpenAI version: {self.version}")
        base_config = OPENAI_SUPPORTED_VERSIONS[self.version]
        return apply_overrides(base_config, config_overrides)

    def _initialize_client(self):
        """Initializes and configures the OpenAI client."""
        # Try to get API key from explicit parameter or environment variable
        try:
            api_key_to_use = self._api_key or get_secret("OPENAI_API_KEY")
        except ValueError:
            # Fall back to default OpenAI credential resolution (env vars, config file)
            log.info("OPENAI_API_KEY not found in environment, using OpenAI's default credential resolution")
            api_key_to_use = None

        if api_key_to_use:
            # Legacy support (might be needed for some setups)
            openai.api_key = api_key_to_use
            return OpenAI(api_key=api_key_to_use)
        else:
            # Let OpenAI SDK handle credential resolution (checks OPENAI_API_KEY env var, config files, etc.)
            return OpenAI()

    def call(
        self,
        query,
        temperature=None,
        seed=None,
        image=None,
        schema=None,
    ):
        if not query:
            raise ValueError("Query cannot be empty")

        # Initial content is just the query
        content = [{"type": "text", "text": query}]
        if image is not None:
            if not isinstance(image, list):
                images = [image]
            else:
                images = image
            for image in images:
                # Encode image and add to content, if image is provided
                base64_image = encode_image(image)
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
                content.append(image_content)

        common_params = {
            "model": self.config.key,
            "messages": [{"role": "user", "content": content}],
            "seed": seed,
        }
        if temperature is not None:
            common_params["temperature"] = temperature

        try:
            # response_format only supported with beta client
            if schema:
                if type(schema) is dict:
                    fix_json_schema(schema)
                    schema = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "my_schema",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                completion = self.client.beta.chat.completions.parse(
                    **common_params, response_format=schema
                )
            else:
                completion = self.client.chat.completions.create(**common_params)

            if not completion or not completion.choices:
                log.error("No completion choices returned")
                return None

            # Extract result
            message = completion.choices[0].message
            if schema:
                if hasattr(message.parsed, "dict"):
                    return message.parsed.dict()
                else:
                    return json.loads(message.content)
            else:
                return message.content.strip()

        except openai.BadRequestError as e:
            log.error(f"Invalid request: {e}")
            raise
        except Exception as e:
            log.error(f"Unexpected error during OpenAI API call: {e}")
            raise

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = None,
        num: int = None,
        image_prompt: list[Image.Image] = None,
        image_prompt_strength: float = 0.5,
        aspect_ratio: str = None,
    ) -> Image.Image:
        # Validate prompt parameter
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Use unified aspect ratio resolution system
        final_width, final_height, final_aspect_ratio = (
            resolve_aspect_ratio_and_dimensions(
                aspect_ratio, self.version, width, height
            )
        )

        # Map the final aspect ratio to OpenAI size string
        openai_size = OPENAI_ASPECT_RATIO_TO_SIZE.get(final_aspect_ratio, "1024x1024")

        log.debug(
            f"OpenAI: Resolved to aspect_ratio '{final_aspect_ratio}' -> dimensions {final_width}x{final_height} -> size '{openai_size}'"
        )

        if image_prompt:
            image_paths = []
            for image in image_prompt:
                image_path = image.info.get("path")
                if not image_path:
                    image_path = save_image_to_disk(image)
                    log.debug(f"Mapping image_prompt = {image_path}")
                image_paths.append(image_path)
            images = [open(path, "rb") for path in image_paths]
            result = self.client.images.edit(
                model="gpt-image-1", image=images, prompt=prompt, size=openai_size
            )
        else:
            result = self.client.images.generate(
                model="gpt-image-1", prompt=prompt, size=openai_size
            )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_bytes))

    def supports_image_editing(self) -> bool:
        """OpenAI supports image editing via images.edit."""
        return True

    def _has_native_edit_image(self) -> bool:
        """OpenAI has native image editing via images.edit API."""
        return False  # We handle editing through generate_image with image_prompt
