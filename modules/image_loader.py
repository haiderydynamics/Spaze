"""Image loading and validation module for SPAZE.

Responsible for safely decoding uploaded floor-plan images, verifying image integrity
against corruption, enforcing dimension and payload limits, and producing a standardized
RGB NumPy image array.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from config.settings import AppConfig
from utils.validation import validate_image_dimensions, ValidationError


class ImageLoadError(Exception):
    """Raised when an uploaded floor-plan image cannot be loaded or validated."""
    pass


# Magic bytes signatures for trusted formats
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


@dataclass(frozen=True)
class LoadedImage:
    """Standardized representation of a validated, decoded floor-plan image."""
    image_rgb: np.ndarray  # uint8 array of shape (H, W, 3) in RGB order
    width: int
    height: int
    channels: int
    format: str
    file_size_bytes: int
    sha256: str
    filename: str

    def __post_init__(self) -> None:
        if self.image_rgb.ndim != 3 or self.image_rgb.shape[2] != 3:
            raise ValueError(f"image_rgb must have shape (H, W, 3), got {self.image_rgb.shape}")


def detect_magic_format(raw_bytes: bytes) -> Optional[str]:
    """Check binary header to detect actual image file format, ignoring extension."""
    if len(raw_bytes) < 8:
        return None
    if raw_bytes.startswith(PNG_MAGIC):
        return "PNG"
    if raw_bytes.startswith(JPEG_MAGIC):
        return "JPEG"
    return None


def load_image_from_bytes(
    raw_bytes: bytes,
    filename: str = "upload.png",
    max_file_size: int = AppConfig.MAX_FILE_SIZE_BYTES,
    min_dim: int = AppConfig.MIN_IMAGE_DIMENSION,
    max_dim: int = AppConfig.MAX_IMAGE_DIMENSION,
) -> LoadedImage:
    """Validate, decode, and standardize an image from raw bytes.

    Args:
        raw_bytes: Raw binary content of the image.
        filename: Original filename (used for extension checking and metadata).
        max_file_size: Maximum allowed file size in bytes.
        min_dim: Minimum width or height in pixels.
        max_dim: Maximum width or height in pixels.

    Returns:
        LoadedImage dataclass with standardized RGB array.

    Raises:
        ImageLoadError: If validation or decoding fails.
    """
    if not raw_bytes:
        raise ImageLoadError("Uploaded file is empty (0 bytes).")

    file_size = len(raw_bytes)
    if file_size > max_file_size:
        raise ImageLoadError(
            f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed "
            f"limit ({max_file_size / (1024*1024):.1f} MB)."
        )

    # Extension verification
    suffix = Path(filename).suffix.lower()
    if suffix not in AppConfig.SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file extension '{suffix}'. Supported formats: {', '.join(AppConfig.SUPPORTED_EXTENSIONS)}"
        )

    # Magic byte header verification (do not trust filename alone)
    detected_format = detect_magic_format(raw_bytes)
    if detected_format is None:
        raise ImageLoadError(
            "File header does not match a valid PNG or JPEG signature. File may be corrupted or disguised."
        )

    # Verify integrity with PIL
    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            pil_img.verify()
    except (UnidentifiedImageError, OSError, Exception) as exc:
        raise ImageLoadError(f"Corrupted or invalid image data: {exc}") from exc

    # Re-open for actual decoding (verify() closes or invalidates the image pointer)
    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            # Handle color conversions: RGBA -> RGB with white background, Gray -> RGB, etc.
            if pil_img.mode in ("RGBA", "LA"):
                # Composite over white background to avoid transparent dark artifacts
                background = Image.new("RGB", pil_img.size, (255, 255, 255))
                alpha = pil_img.split()[-1]
                background.paste(pil_img, mask=alpha)
                rgb_img = background
            elif pil_img.mode == "P":
                # Palette image
                rgb_img = pil_img.convert("RGB")
            elif pil_img.mode == "L":
                # Grayscale
                rgb_img = pil_img.convert("RGB")
            elif pil_img.mode == "RGB":
                rgb_img = pil_img.copy()
            else:
                rgb_img = pil_img.convert("RGB")

            width, height = rgb_img.size

            # Dimension boundary verification
            try:
                validate_image_dimensions(
                    width=width,
                    height=height,
                    channels=3,
                    min_dim=min_dim,
                    max_dim=max_dim,
                )
            except ValidationError as val_err:
                raise ImageLoadError(str(val_err)) from val_err

            # Convert to uint8 NumPy array
            image_array = np.array(rgb_img, dtype=np.uint8)

    except ImageLoadError:
        raise
    except Exception as exc:
        raise ImageLoadError(f"Failed to decode image: {exc}") from exc

    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    return LoadedImage(
        image_rgb=image_array,
        width=width,
        height=height,
        channels=3,
        format=detected_format,
        file_size_bytes=file_size,
        sha256=sha256_hash,
        filename=Path(filename).name,
    )


def load_image_from_path(
    file_path: Union[str, Path],
    max_file_size: int = AppConfig.MAX_FILE_SIZE_BYTES,
) -> LoadedImage:
    """Load and validate an image directly from a local filesystem path."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise ImageLoadError(f"Image file does not exist at path: {path}")

    try:
        raw_bytes = path.read_bytes()
    except Exception as exc:
        raise ImageLoadError(f"Unable to read file at {path}: {exc}") from exc

    return load_image_from_bytes(
        raw_bytes=raw_bytes,
        filename=path.name,
        max_file_size=max_file_size,
    )
