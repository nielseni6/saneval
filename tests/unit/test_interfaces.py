"""Unit tests for ssa.interfaces module."""

from PIL import Image

from ssa.interfaces import ImageInfo


class TestImageInfo:
    """Tests for ImageInfo dataclass."""

    def test_imageinfo_creation(self, mock_image):
        """ImageInfo should be created correctly."""
        img_info = ImageInfo(
            path="/path/to/image.png", image_id="test-001", pil_image=mock_image
        )
        assert img_info.path == "/path/to/image.png"
        assert img_info.image_id == "test-001"
        assert isinstance(img_info.pil_image, Image.Image)

    def test_imageinfo_info_property(self, mock_image):
        """Info property should return dict."""
        img_info = ImageInfo(path="/test.png", image_id="id-1", pil_image=mock_image)
        info = img_info.info
        assert isinstance(info, dict)
        assert info["path"] == "/test.png"
        assert info["image_id"] == "id-1"

    def test_imageinfo_with_different_paths(self, mock_image):
        """ImageInfo should handle various path formats."""
        paths = [
            "/absolute/path/image.png",
            "relative/path/image.jpg",
            "image.webp",
        ]
        for path in paths:
            img_info = ImageInfo(path=path, image_id="test", pil_image=mock_image)
            assert img_info.path == path

    def test_imageinfo_with_complex_id(self, mock_image):
        """ImageInfo should handle complex image IDs."""
        complex_ids = [
            "test-001",
            "prompt-abc123xyz_0",
            "benchmark-spatial-001",
        ]
        for img_id in complex_ids:
            img_info = ImageInfo(
                path="/test.png", image_id=img_id, pil_image=mock_image
            )
            assert img_info.image_id == img_id
