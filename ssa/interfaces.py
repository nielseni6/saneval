from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from PIL import Image

if TYPE_CHECKING:
    from ssa.scorers.model_scorer import ModelScorer

from ssa.prompts import Corpus, Prompt  # Added Corpus

# Image type can be either a PIL Image or a string path
ImageType = Union[Image.Image, str]


@dataclass
class ImageInfo:
    """
    Standardized container for image data with metadata.

    Attributes:
        path: File system path to the image
        image_id: Unique identifier for the image
        pil_image: PIL Image object
    """
    path: str
    image_id: str
    pil_image: Image.Image

    @property
    def info(self) -> Dict[str, str]:
        """Return info dict for compatibility with legacy code."""
        return {"path": self.path, "image_id": self.image_id}


class ImageEditingCapability(ABC):
    """Abstract base class for image editing capabilities."""

    @abstractmethod
    def supports_image_editing(self) -> bool:
        """Return True if the provider supports image editing operations."""
        pass

    @abstractmethod
    def edit_image(
        self, prompt: str, image: ImageType, **kwargs: Any
    ) -> Optional[ImageType]:
        """
        Edit an image based on a text prompt.

        Args:
            prompt: Text prompt describing the desired changes
            image: Input image to edit
            **kwargs: Additional editing parameters

        Returns:
            Edited image or None if editing failed
        """
        pass


class BaseImageEditingMixin(ImageEditingCapability):
    """
    Base mixin class providing default image editing behavior.

    This can be used by providers that don't have dedicated editing methods
    but can perform editing through their generate_image method.
    """

    def supports_image_editing(self) -> bool:
        """
        Check if this provider supports image editing.

        Default implementation checks for either edit_image method
        or image_prompt parameter support in generate_image.
        """
        # Check for dedicated edit_image method
        if hasattr(self, "_has_native_edit_image") and self._has_native_edit_image():
            return True

        # Check for image_prompt support in generate_image
        if hasattr(self, "generate_image"):
            import inspect

            sig = inspect.signature(self.generate_image)
            return any(param in sig.parameters for param in ["image_prompt", "image"])

        return False

    def edit_image(
        self,
        prompt: str,
        image: ImageType,
        guidance_scale: Optional[float] = None,
        image_prompt_strength: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[ImageType]:
        """
        Default image editing implementation using generate_image.

        This method provides a fallback for providers that don't have
        a dedicated edit_image method but support image prompts.
        """
        # Check if the provider supports image editing
        if not self.supports_image_editing():
            raise NotImplementedError(
                f"Provider {self.__class__.__name__} does not support image editing"
            )

        # If provider has native edit_image, delegate to it
        if hasattr(self, "_has_native_edit_image") and self._has_native_edit_image():
            if hasattr(self, "_native_edit_image"):
                return self._native_edit_image(prompt, image, **kwargs)

        # Fallback to generate_image with image prompt
        if hasattr(self, "generate_image"):
            import inspect

            sig = inspect.signature(self.generate_image)

            # Prepare parameters based on what the provider accepts
            edit_kwargs = kwargs.copy()

            if "image_prompt" in sig.parameters:
                edit_kwargs["image_prompt"] = [image]
                if guidance_scale is not None:
                    edit_kwargs["image_prompt_strength"] = guidance_scale
                elif image_prompt_strength is not None:
                    edit_kwargs["image_prompt_strength"] = image_prompt_strength
            elif "image" in sig.parameters:
                edit_kwargs["image"] = image
                if guidance_scale is not None:
                    edit_kwargs["guidance_scale"] = guidance_scale
            else:
                # Provider doesn't support image editing
                raise NotImplementedError(
                    f"Provider {self.__class__.__name__} does not support image editing"
                )

            # Add aspect_ratio if the provider's generate_image supports it
            if "aspect_ratio" in sig.parameters and aspect_ratio is not None:
                edit_kwargs["aspect_ratio"] = aspect_ratio

            return self.generate_image(prompt, **edit_kwargs)

        raise NotImplementedError(
            f"Provider {self.__class__.__name__} does not support image editing"
        )


def normalize_edit_parameters(
    guidance_scale: Optional[float] = None,
    image_prompt_strength: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Normalize editing parameters across different provider interfaces.

    Args:
        guidance_scale: Guidance scale for editing (Imagen-style)
        image_prompt_strength: Image prompt strength (Replicate-style)
        **kwargs: Additional parameters

    Returns:
        Normalized parameter dict
    """
    normalized = kwargs.copy()

    # Use guidance_scale if provided, otherwise fall back to image_prompt_strength
    strength = guidance_scale if guidance_scale is not None else image_prompt_strength
    if strength is not None:
        normalized["strength"] = strength

    return normalized


class BenchmarkScorer(ABC):
    """
    Abstract Base Class for benchmark scorers.
    Each specific scorer (DSG, FaceAnalysis, etc.) should implement this interface.
    """

    def __init__(
        self,
        scorer_key: str,
        model_scorer: "ModelScorer",
        corpus_prompts: Optional[List[Prompt]] = None,
        corpus: Optional[Corpus] = None,  # Added corpus
    ):
        self.scorer_key = scorer_key
        self.model_scorer = model_scorer  # The underlying ssa.scorers.model_scorer.ModelScorer instance from ssa.model
        self.corpus_prompts = (
            corpus_prompts  # For initializers that need full corpus info
        )
        self.corpus = corpus  # Store corpus instance
        # Initialize aggregators, passing corpus and prompts if available
        self.aggregators: Dict[str, Any] = self._initialize_aggregators_internal(
            corpus=self.corpus, corpus_prompts=self.corpus_prompts
        )

    @abstractmethod
    def _initialize_aggregators_internal(
        self,
        corpus: Optional[Corpus] = None,  # Added corpus
        corpus_prompts: Optional[List[Prompt]] = None,  # Added corpus_prompts
    ) -> Dict[str, Any]:
        """
        Initializes and returns a dictionary of aggregators specific to this scorer.
        The keys are aggregator names and values are Aggregator instances or dicts of Aggregators.
        This method can use self.corpus or self.corpus_prompts if needed.
        """
        pass

    @abstractmethod
    def score_prompt(
        self,
        prompt: Prompt,
        image: Optional[ImageType] = None,  # Made image optional
        image_path: Optional[str] = None,  # Added image_path for scorers that use it
        prompt_result: Optional[
            Dict[str, Any]
        ] = None,  # Allow passing prompt_result dict
        running_metrics: Optional[
            Dict[str, Any]
        ] = None,  # Allow passing running_metrics dict
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:  # Return updated prompt_result or new dict
        """
        Scores a single prompt.
        Implementations should handle image/image_path based on their needs.
        Should update prompt_result with scoring information under self.scorer_key.
        Can optionally update running_metrics.
        Returns the dictionary that was passed as prompt_result, or a new one if None was passed.
        """
        pass

    @abstractmethod
    def aggregate_metrics(
        self, final_agg_dict: Dict[str, Any]
    ) -> Dict[str, Any]:  # Return updated final_agg_dict
        """
        Aggregates metrics for the full run and updates final_agg_dict.
        Returns the dictionary that was passed as final_agg_dict.
        """
        pass

    # Optional warmup method
    def warmup(self) -> None:
        """
        Optional method for scorers to perform any necessary warmup,
        like loading models or pre-calculating values.
        This method will be called once before any scoring begins.
        """
        pass

    # Helper method to get initialized aggregators, can be overridden if complex logic needed
    def get_aggregators(self) -> Dict[str, Any]:
        return self.aggregators
