import base64
import math
import os
import random
import tempfile
from io import BytesIO
from textwrap import wrap

import requests
from cog import Path
from PIL import Image, ImageDraw, ImageFont

from ssa.utils.base import repo_base
from ssa.utils.fonts import DEFAULT_BODY_FONT, DEFAULT_WATERMARK_FONT

test_image_path = repo_base / "tests" / "test_image.png"


def watermark_image(image, font_size=16):
    w, h = image.size
    watermarked_image = image.copy()
    draw = ImageDraw.Draw(watermarked_image)
    font = ImageFont.truetype(DEFAULT_WATERMARK_FONT, int(font_size))
    draw.text(
        (w - 64, h - 10), "© 2ND SET AI", font=font, fill=(150, 150, 150), anchor="ms"
    )
    return draw._image


def image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    return Image.open(BytesIO(response.content))


def image_info(image):
    return {k: v for k, v in image.info.items() if not k.startswith("jf")}


def save_image_to_disk(image: Image.Image, format="png"):
    output_path = Path(tempfile.mkdtemp()) / f"processed.{format}"
    image.save(output_path, format)
    return Path(output_path)


def round_to(x, allowed):
    for idx, y in enumerate(allowed):
        if y > x:
            return y


def load_image(reference):
    """Load a PIL image from a local path"""
    if os.path.exists(reference):
        img = Image.open(reference)
        img.info["path"] = reference
        return _handle_special_formats(img, reference)
    else:
        raise ValueError(f"Unrecognized image : {reference}")


def _handle_special_formats(img, reference):
    """Handle special image formats like MPO by converting to standard formats"""
    # Handle MPO (Multi-Picture Object) files - convert to JPEG
    if hasattr(img, "format") and img.format == "MPO":
        # MPO files contain multiple images, we'll use the first one
        # Convert to RGB mode to ensure JPEG compatibility
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Create a new image to remove MPO-specific metadata that causes issues
        new_img = Image.new("RGB", img.size)
        new_img.paste(img)
        new_img.info["path"] = reference
        new_img.info["original_format"] = "MPO"
        return new_img

    return img


def get_image_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()


def get_image_html(image_path, width=300):
    """Render html for image path"""
    if not os.path.exists(image_path):
        return f"No Image Found: {image_path}"
    image_b64 = get_image_base64(image_path)
    return f'<img src="data:image/png;base64,{image_b64}" width={width}"/><br>'


def draw_mock_image(text="", height=800, width=800, num_circles=10, add_text=True):
    image = Image.new("RGB", (height, width), color="white")
    draw = ImageDraw.Draw(image)

    # Draw some random shapes
    for _ in range(num_circles):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(100, 400)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        draw.ellipse([x, y, x + size, y + size], fill=color)

    # Add prompt text for debuggability
    if add_text:
        lines = wrap(f"Is this a {text}", width=60)
        try:
            font = ImageFont.truetype(DEFAULT_BODY_FONT, 32)
        except (OSError, IOError):
            font = ImageFont.load_default()
            # Default font size handling

        text_y = 50
        # Calculate line height dynamically based on font metrics
        line_height_bbox = font.getbbox("Ay")
        line_height = line_height_bbox[3] - line_height_bbox[1] + 2
        for line in lines:
            draw.text((50, text_y), line, font=font, fill="black")
            text_y += line_height

    return image


def resize_image(img, size):
    """Resize image to given size but Preserve aspect ratio"""
    width, height = img.size
    aspect_ratio = width / height

    if width > height:
        new_width = size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = size[1]
        new_width = int(new_height * aspect_ratio)

    # Ensure dimensions don't exceed the target size
    new_width = min(new_width, size[0])
    new_height = min(new_height, size[1])

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def encode_image(image: Image.Image) -> str:
    """Returns base64 encoding string for an image"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_image_path(image_path: str) -> str:
    """Returns base64 encoding string for an image_path"""
    if image_path == "test" or image_path == "default":
        image_path = str(test_image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode("utf-8")


def create_combined_grid_image(
    image_paths: list[str],
    prompts: list[str],
    max_image_size: tuple = (400, 400),
    padding: int = 20,
    font_size: int = 14,
    bg_color: str = "white",
) -> Image.Image:
    """
    Create a combined grid image with generated images and their prompts below each image.

    Args:
        image_paths: List of paths to generated images
        prompts: List of prompts corresponding to each image
        max_image_size: Maximum size for each image in the grid (width, height)
        padding: Padding between images and text
        font_size: Font size for prompt text
        bg_color: Background color for the combined image

    Returns:
        PIL Image with the combined grid layout
    """
    if not image_paths or not prompts or len(image_paths) != len(prompts):
        raise ValueError(
            "image_paths and prompts must be non-empty and have the same length"
        )

    num_images = len(image_paths)

    # Determine grid layout based on number of images
    if num_images == 1:
        grid_cols, grid_rows = 1, 1
    elif num_images == 2:
        grid_cols, grid_rows = 2, 1
    elif num_images == 3:
        grid_cols, grid_rows = 3, 1
    elif num_images == 4:
        grid_cols, grid_rows = 2, 2
    else:
        # For larger sets, try to create a roughly square grid
        grid_cols = math.ceil(math.sqrt(num_images))
        grid_rows = math.ceil(num_images / grid_cols)

    # Load and resize images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            img = resize_image(img, max_image_size)
            images.append(img)
        except Exception:
            # Create a placeholder image if loading fails
            placeholder = Image.new("RGB", max_image_size, color="lightgray")
            draw = ImageDraw.Draw(placeholder)
            draw.text(
                (max_image_size[0] // 2, max_image_size[1] // 2),
                "Error loading image",
                fill="black",
                anchor="mm",
            )
            images.append(placeholder)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype(DEFAULT_BODY_FONT, font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Arial.ttf", font_size
            )  # macOS
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Calculate dimensions for text areas
    max_text_width = max_image_size[0]
    text_heights = []
    wrapped_prompts = []

    # Cache font measurements to avoid repeated calls
    line_height_bbox = font.getbbox("Ay")
    line_height = line_height_bbox[3] - line_height_bbox[1] + 2
    font_bbox_cache = {}  # Cache for text width measurements

    for prompt in prompts:
        # Wrap text to fit within image width
        words = prompt.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word

            # Use cache for text width measurements
            if test_line not in font_bbox_cache:
                bbox = font.getbbox(test_line)
                font_bbox_cache[test_line] = bbox[2] - bbox[0]
            text_width = font_bbox_cache[test_line]

            if text_width <= max_text_width - padding * 2:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        wrapped_prompts.append(lines)

        # Calculate text height for this prompt using cached line height
        text_height = len(lines) * line_height + padding
        text_heights.append(text_height)

    # Calculate cell dimensions
    cell_width = max_image_size[0] + padding * 2
    cell_height = max_image_size[1] + max(text_heights) + padding * 3

    # Calculate total canvas size
    canvas_width = grid_cols * cell_width
    canvas_height = grid_rows * cell_height

    # Create the combined image
    combined_image = Image.new("RGB", (canvas_width, canvas_height), color=bg_color)

    # Place images and text
    for i, (img, wrapped_text) in enumerate(zip(images, wrapped_prompts)):
        row = i // grid_cols
        col = i % grid_cols

        # Calculate position
        x = col * cell_width + padding
        y = row * cell_height + padding

        # Paste the image
        img_x = x + (max_image_size[0] - img.width) // 2
        img_y = y
        combined_image.paste(img, (img_x, img_y))

        # Draw the text below the image
        text_y = y + max_image_size[1] + padding
        draw = ImageDraw.Draw(combined_image)

        for line in wrapped_text:
            # Use cached measurement or calculate if not cached
            if line not in font_bbox_cache:
                bbox = font.getbbox(line)
                font_bbox_cache[line] = bbox[2] - bbox[0]
            text_width = font_bbox_cache[line]

            text_x = x + (max_image_size[0] - text_width) // 2
            draw.text((text_x, text_y), line, font=font, fill="black")
            text_y += line_height

    return combined_image
