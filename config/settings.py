"""Centralized configuration for the SPAZE architectural floor-plan processing system.

Defines all constants, thresholds, supported formats, and parameter defaults
to prevent magic numbers across the codebase.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class AppConfig:
    """General application metadata and upload constraints."""

    APP_NAME: str = "SPAZE"
    TAGLINE: str = "Imagine. Generate. Inhabit."
    PHASE: str = "Phase 1: Foundation & Structured Room Extraction"
    VERSION: str = "0.1.0-phase1"

    # Supported formats and size constraints
    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
    SUPPORTED_MIME_TYPES: Tuple[str, ...] = ("image/png", "image/jpeg")
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB max payload
    MAX_IMAGE_DIMENSION: int = 4096  # Maximum width or height in pixels
    MIN_IMAGE_DIMENSION: int = 64    # Minimum width or height in pixels


@dataclass(frozen=True)
class ProcessingConfig:
    """Parameters for computer vision and image processing pipelines."""

    # Denoising
    GAUSSIAN_BLUR_KERNEL: Tuple[int, int] = (5, 5)
    GAUSSIAN_BLUR_SIGMA: float = 1.0
    BILATERAL_D: int = 9
    BILATERAL_SIGMA_COLOR: float = 75.0
    BILATERAL_SIGMA_SPACE: float = 75.0

    # Thresholding
    ADAPTIVE_METHOD: str = "gaussian"  # 'gaussian', 'mean', or 'otsu'
    ADAPTIVE_BLOCK_SIZE: int = 15      # Must be odd and > 1
    ADAPTIVE_C: float = 4.0

    # Edge Detection
    CANNY_THRESHOLD1: float = 50.0
    CANNY_THRESHOLD2: float = 150.0
    CANNY_APERTURE_SIZE: int = 3

    # Morphological Operations for Wall Enhancement
    WALL_MORPH_KERNEL_SIZE: int = 3
    WALL_CLOSING_ITERATIONS: int = 2

    # Hough Transform for Wall Line Candidates
    HOUGH_RHO: float = 1.0
    HOUGH_THETA_DEGREES: float = 1.0
    HOUGH_THRESHOLD: int = 60
    HOUGH_MIN_LINE_LENGTH: int = 40
    HOUGH_MAX_LINE_GAP: int = 10
    HV_ANGLE_TOLERANCE_DEGREES: float = 12.0  # Tolerance around 0, 90, 180 degrees


@dataclass(frozen=True)
class RoomConfig:
    """Parameters and categories for room candidate extraction and classification."""

    # Allowed architectural room types
    ROOM_TYPES: Tuple[str, ...] = (
        "Living Room",
        "Bedroom",
        "Kitchen",
        "Bathroom",
        "Dining Room",
        "Hallway",
        "Balcony",
        "Utility Room",
        "Other",
        "Unclassified",
    )

    # Allowed detection status types
    DETECTION_STATUSES: Tuple[str, ...] = (
        "Candidate",
        "Needs Review",
        "User Confirmed",
        "Rejected",
    )

    # Filtering thresholds
    MIN_ROOM_AREA_PIXELS: float = 600.0       # Minimum enclosed area in pixels
    MIN_ROOM_AREA_RATIO: float = 0.003        # Area must be >= 0.3% of total image
    MAX_ROOM_AREA_RATIO: float = 0.85         # Area must be <= 85% of total image
    MIN_ASPECT_RATIO: float = 0.12            # Reject extreme sliver lines
    MAX_ASPECT_RATIO: float = 8.5
    POLYGON_APPROX_EPSILON_RATIO: float = 0.02  # Douglas-Peucker epsilon factor
    MIN_POLYGON_VERTICES: int = 4
    MAX_POLYGON_VERTICES: int = 20

    # Heuristic Quality Scoring Weights
    RECTANGULARITY_WEIGHT: float = 0.45
    SOLIDITY_WEIGHT: float = 0.35
    ASPECT_RATIO_WEIGHT: float = 0.20
