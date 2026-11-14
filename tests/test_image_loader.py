"""Unit tests for modules.image_loader."""
import io
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from modules.image_loader import (
    ImageLoadError,
    LoadedImage,
    detect_magic_format,
    load_image_from_bytes,
    load_image_from_path,
)


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Generate in-memory valid PNG image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (200, 150), color=(255, 255, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Generate in-memory valid JPEG image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (200, 150), color=(200, 200, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_load_valid_png(valid_png_bytes):
    loaded = load_image_from_bytes(valid_png_bytes, filename="test_plan.png")
    assert isinstance(loaded, LoadedImage)
    assert loaded.width == 200
    assert loaded.height == 150
    assert loaded.channels == 3
    assert loaded.format == "PNG"
    assert loaded.image_rgb.shape == (150, 200, 3)
    assert len(loaded.sha256) == 64


def test_load_valid_jpeg(valid_jpeg_bytes):
    loaded = load_image_from_bytes(valid_jpeg_bytes, filename="test_plan.jpg")
    assert isinstance(loaded, LoadedImage)
    assert loaded.width == 200
    assert loaded.height == 150
    assert loaded.format == "JPEG"


def test_load_empty_file():
    with pytest.raises(ImageLoadError, match="empty"):
        load_image_from_bytes(b"", filename="empty.png")


def test_unsupported_file_extension(valid_png_bytes):
    with pytest.raises(ImageLoadError, match="Unsupported file extension"):
        load_image_from_bytes(valid_png_bytes, filename="plan.gif")


def test_disguised_invalid_bytes():
    # File named .png but containing plain text or random garbage
    fake_bytes = b"This is not a real image file."
    with pytest.raises(ImageLoadError, match="valid PNG or JPEG signature"):
        load_image_from_bytes(fake_bytes, filename="fake.png")


def test_corrupted_png_header():
    # Starts with PNG magic but corrupted thereafter
    corrupted = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00CORRUPT_DATA"
    with pytest.raises(ImageLoadError, match="Corrupted or invalid image data"):
        load_image_from_bytes(corrupted, filename="corrupt.png")


def test_excessive_dimensions():
    # Create image exceeding max_dim limit
    buf = io.BytesIO()
    img = Image.new("RGB", (5000, 100), color=(255, 255, 255))
    img.save(buf, format="PNG")
    huge_bytes = buf.getvalue()

    with pytest.raises(ImageLoadError, match="exceed maximum allowed dimension"):
        load_image_from_bytes(huge_bytes, filename="huge.png", max_dim=4096)


def test_undersized_dimensions():
    # Create image smaller than min_dim
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    img.save(buf, format="PNG")
    small_bytes = buf.getvalue()

    with pytest.raises(ImageLoadError, match="smaller than minimum allowed"):
        load_image_from_bytes(small_bytes, filename="tiny.png", min_dim=64)


def test_excessive_file_size(valid_png_bytes):
    # Set max_file_size smaller than fixture
    with pytest.raises(ImageLoadError, match="exceeds maximum allowed"):
        load_image_from_bytes(valid_png_bytes, filename="plan.png", max_file_size=10)


def test_load_from_path(tmp_path, valid_png_bytes):
    file_path = tmp_path / "sample.png"
    file_path.write_bytes(valid_png_bytes)

    loaded = load_image_from_path(file_path)
    assert loaded.width == 200
    assert loaded.filename == "sample.png"


def test_load_nonexistent_path():
    with pytest.raises(ImageLoadError, match="does not exist"):
        load_image_from_path("nonexistent_floorplan_12345.png")
