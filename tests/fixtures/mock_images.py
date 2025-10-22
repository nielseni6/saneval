"""Helper functions for creating mock test images."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_simple_test_image(width=100, height=100, color=(255, 0, 0)):
    """
    Create a simple solid color test image.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple

    Returns:
        PIL Image object
    """
    return Image.new("RGB", (width, height), color)


def create_test_image_with_shapes(width=200, height=200):
    """
    Create test image with basic shapes.

    Creates an image with a red circle and blue rectangle
    for testing shape detection and spatial reasoning.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        PIL Image object with shapes
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Draw a red circle
    draw.ellipse([50, 50, 100, 100], fill="red", outline="black")

    # Draw a blue rectangle
    draw.rectangle([120, 50, 170, 100], fill="blue", outline="black")

    return img


def create_test_image_with_objects(num_objects=3, width=300, height=200, colors=None):
    """
    Create test image with multiple colored circles.

    Useful for testing numeracy and counting scorers.

    Args:
        num_objects: Number of objects to draw
        width: Image width in pixels
        height: Image height in pixels
        colors: List of RGB color tuples (defaults to random)

    Returns:
        PIL Image object with multiple objects
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    if colors is None:
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

    spacing = width // (num_objects + 1)
    radius = min(30, spacing // 2)

    for i in range(num_objects):
        x = spacing * (i + 1)
        y = height // 2
        color = colors[i % len(colors)]

        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline="black",
        )

    return img


def create_test_image_with_text(
    text="Test", width=200, height=100, bg_color="white", text_color="black"
):
    """
    Create test image with text.

    Args:
        text: Text to render
        width: Image width in pixels
        height: Image height in pixels
        bg_color: Background color
        text_color: Text color

    Returns:
        PIL Image object with text
    """
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Use default font (PIL may not have truetype fonts)
    try:
        # Try to center text
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((width - text_width) // 2, (height - text_height) // 2)
        draw.text(position, text, fill=text_color)
    except:
        # Fallback if text measurement fails
        draw.text((width // 4, height // 3), text, fill=text_color)

    return img


def create_random_noise_image(width=100, height=100):
    """
    Create an image with random noise.

    Useful for testing robustness.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        PIL Image with random RGB values
    """
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)
