"""OpenCV Preprocessing Pipeline for Floor-Plan Images.

Executes genuine, deterministic computer vision operations:
1. Standard RGB-to-Grayscale conversion
2. Edge-preserving noise reduction (Bilateral or Gaussian)
3. Adaptive / Otsu Thresholding (Binarization)
4. Canny Edge Detection
5. Morphological Closing and Wall Enhancement

All intermediate stages are retained without modifying the original source image.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from config.settings import ProcessingConfig


@dataclass(frozen=True)
class ProcessedStages:
    """Immutable bundle of all intermediate computer vision processing outputs."""
    original_rgb: np.ndarray
    grayscale: np.ndarray
    denoised: np.ndarray
    threshold: np.ndarray
    edges: np.ndarray
    morphology: np.ndarray
    parameters: Dict[str, Any]
    processing_time_ms: float


class ImageProcessor:
    """Coordinates deterministic OpenCV operations for floor-plan images."""

    def __init__(self, config: Optional[ProcessingConfig] = None) -> None:
        self.config = config or ProcessingConfig()

    def process(
        self,
        image_rgb: np.ndarray,
        use_bilateral: bool = False,
        adaptive_method: Optional[str] = None,
        canny_th1: Optional[float] = None,
        canny_th2: Optional[float] = None,
    ) -> ProcessedStages:
        """Execute the full computer vision preprocessing pipeline.

        Args:
            image_rgb: Input image in RGB uint8 format (H, W, 3).
            use_bilateral: If True, uses bilateral filter; otherwise Gaussian blur.
            adaptive_method: 'gaussian', 'mean', or 'otsu'. Defaults to config.
            canny_th1: Canny edge lower threshold. Defaults to config.
            canny_th2: Canny edge upper threshold. Defaults to config.

        Returns:
            ProcessedStages dataclass containing all intermediate maps.
        """
        start_time = time.perf_counter()

        # Ensure we never mutate the input
        src_rgb = np.copy(image_rgb)

        # 1. Grayscale Conversion
        gray = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)

        # 2. Noise Reduction
        if use_bilateral:
            denoised = cv2.bilateralFilter(
                gray,
                d=self.config.BILATERAL_D,
                sigmaColor=self.config.BILATERAL_SIGMA_COLOR,
                sigmaSpace=self.config.BILATERAL_SIGMA_SPACE,
            )
        else:
            denoised = cv2.GaussianBlur(
                gray,
                ksize=self.config.GAUSSIAN_BLUR_KERNEL,
                sigmaX=self.config.GAUSSIAN_BLUR_SIGMA,
            )

        # 3. Thresholding (Binarization)
        # In architectural plans, walls are typically dark on a light background.
        # We invert so walls/lines become 255 (foreground) and rooms/floors become 0 (background).
        method = (adaptive_method or self.config.ADAPTIVE_METHOD).lower()
        if method == "otsu":
            _, thresh = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        elif method == "mean":
            thresh = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                self.config.ADAPTIVE_BLOCK_SIZE,
                self.config.ADAPTIVE_C,
            )
        else:  # 'gaussian' default
            thresh = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.config.ADAPTIVE_BLOCK_SIZE,
                self.config.ADAPTIVE_C,
            )

        # 4. Canny Edge Detection
        th1 = canny_th1 if canny_th1 is not None else self.config.CANNY_THRESHOLD1
        th2 = canny_th2 if canny_th2 is not None else self.config.CANNY_THRESHOLD2
        edges = cv2.Canny(
            denoised,
            threshold1=th1,
            threshold2=th2,
            apertureSize=self.config.CANNY_APERTURE_SIZE,
        )

        # 5. Morphological Wall Enhancement (Closing small gaps to form solid enclosed boundaries)
        k_size = self.config.WALL_MORPH_KERNEL_SIZE
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        # Combine thresholded lines and edges to create a solid wall mask
        combined_walls = cv2.bitwise_or(thresh, edges)
        morphology = cv2.morphologyEx(
            combined_walls,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.WALL_CLOSING_ITERATIONS,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        parameters = {
            "noise_filter": "bilateral" if use_bilateral else "gaussian_blur",
            "gaussian_kernel": self.config.GAUSSIAN_BLUR_KERNEL,
            "threshold_method": method,
            "canny_thresholds": (th1, th2),
            "morph_kernel_size": k_size,
            "morph_iterations": self.config.WALL_CLOSING_ITERATIONS,
        }

        return ProcessedStages(
            original_rgb=src_rgb,
            grayscale=gray,
            denoised=denoised,
            threshold=thresh,
            edges=edges,
            morphology=morphology,
            parameters=parameters,
            processing_time_ms=round(elapsed_ms, 2),
        )
