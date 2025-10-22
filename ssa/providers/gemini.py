import json

from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types as genaitypes
from pydantic import BaseModel

from ssa.schemas import validate_response_against_schema
from ssa.utils.logging import get_log
from ssa.utils.secrets import get_secret
from ssa.utils.system import apply_overrides

log = get_log(__file__)

# Gemini VLM/LLM Models
GEMINI_2_5_FLASH_LITE_PREV = "gemini/2.5-flash-lite-preview"
GEMINI_2_5_FLASH = "gemini/2.5-flash"
GEMINI_2_5_PRO = "gemini/2.5-pro"


class GeminiConfig(BaseModel):
    key: str = ""
    max_retries: int = 7
    temperature: float = 0.5


# see https://ai.google.dev/pricing

DEFAULT_VERSION = GEMINI_2_5_FLASH
SUPPORTED_VERSIONS = {
    GEMINI_2_5_FLASH_LITE_PREV: GeminiConfig(
        key="gemini-2.5-flash-lite-preview-06-17",
    ),
    GEMINI_2_5_FLASH: GeminiConfig(key="gemini-2.5-flash"),
    GEMINI_2_5_PRO: GeminiConfig(key="gemini-2.5-pro"),
}


class GeminiProvider:
    def __init__(self, version=DEFAULT_VERSION, config_overrides=None):
        self.version = version
        # Initialize the google.genai client
        # Try to get API key from environment, otherwise use application default credentials
        try:
            api_key = get_secret("GOOGLE_API_KEY")
            self.client = genai.Client(api_key=api_key)
        except ValueError:
            # Fall back to application default credentials (gcloud auth application-default login)
            log.info("GOOGLE_API_KEY not found, using application default credentials")
            self.client = genai.Client()

        self.config = apply_overrides(SUPPORTED_VERSIONS[version], config_overrides)
        self.max_retries = self.config.max_retries

    def call(
        self,
        query,
        image=None,
        schema=None,
        seed=None,
        temperature=1.0,
        include_thoughts=False,
        thinking_budget=-1,
        debug_logs=False,
    ):
        """Generate content using the Gemini model.

        Args:
            query: The text prompt/query to send to the model
            image: Optional image(s) to include with the query. Can be a single image or list of images
            schema: Optional JSON schema for structured output responses
            seed: Random seed for reproducible outputs (not yet supported by Gemini)
            temperature: Controls randomness in generation (0.0-2.0, default 1.0)
            include_thoughts: If True, includes the model's internal reasoning process in the response.
                This enables chain-of-thought reasoning for better quality outputs at the cost of additional tokens.
            thinking_budget: Maximum number of tokens the model can use for internal reasoning (-1 for unlimited).
                Higher budgets allow more thorough reasoning
                but consume more tokens and increase latency.
            debug_logs: If True, logs detailed token usage statistics and reasoning thoughts to debug level.
                Useful for monitoring token consumption and understanding model reasoning patterns.

        Returns:
            Generated text response from the model

        Raises:
            NotImplementedError: If seed parameter is provided (not yet supported by Gemini)
        """
        response = ""
        if image:
            if isinstance(image, list):
                payload = image + [query]
            else:
                payload = [image, query]
        else:
            payload = [query]

        if seed:
            raise NotImplementedError(
                "genai does not support seed yet, see https://github.com/2ndSetAI/code/issues/15"
            )

        kwargs = {
            "temperature": temperature,  # 0.0-2.0, 1.0 default
        }
        if schema:
            kwargs.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                }
            )

        # Add thinking config for reasoning models
        kwargs["thinking_config"] = genaitypes.ThinkingConfig(
            include_thoughts=include_thoughts, thinking_budget=thinking_budget
        )

        # Add safety settings
        kwargs["safety_settings"] = [
            genaitypes.SafetySetting(
                category=genaitypes.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=genaitypes.HarmBlockThreshold.BLOCK_NONE,
            ),
            genaitypes.SafetySetting(
                category=genaitypes.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=genaitypes.HarmBlockThreshold.BLOCK_NONE,
            ),
            genaitypes.SafetySetting(
                category=genaitypes.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=genaitypes.HarmBlockThreshold.BLOCK_NONE,
            ),
            genaitypes.SafetySetting(
                category=genaitypes.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=genaitypes.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        generation_config = genaitypes.GenerateContentConfig(**kwargs)

        try:

            response = self.client.models.generate_content(
                model=self.config.key,
                contents=payload,
                config=generation_config,
            )
            if (
                debug_logs
                and hasattr(response, "usage_metadata")
                and response.usage_metadata
            ):
                usage = response.usage_metadata
                log.debug(
                    f"Token usage - Candidates: {usage.candidates_token_count}, Prompt: {usage.prompt_token_count}, Thoughts: {getattr(usage, 'thoughts_token_count', 0)}, Total: {usage.total_token_count}"
                )
                try:
                    for part in response.candidates[0].content.parts:
                        if not part.text:
                            continue
                        elif part.thought:
                            log.debug(f"Response thoughts: {part.text}")
                except (AttributeError, TypeError) as e:
                    log.warning(
                        f"Could not properly read gemini response candidates content parts. Received error: {e}"
                    )
        except google_exceptions.InvalidArgument:
            log.error(f"Invalid argument, payload={payload} config={generation_config}")
            raise

        text = response.text
        if schema:
            try:
                js = json.loads(text)
            except json.JSONDecodeError:
                log.error(f"Bad response Json: {text}")
                raise
            validate_response_against_schema(js, schema)
            return js
        return text
