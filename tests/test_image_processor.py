"""Unit tests for modules.image_processor."""
import numpy as np
import pytest

from modules.image_processor import ImageProcessor, ProcessedStages


@pytest.fixture
def test_floorplan_rgb() -> np.ndarray:
    """Create a synthetic 100x100 RGB image with a black wall rectangle on white background."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # Draw dark rectangle (walls)
    img[20:80, 20:25] = 0   # Left wall
    img[20:80, 75:80] = 0   # Right wall
    img[20:25, 20:80] = 0   # Top wall
    img[75:80, 20:80] = 0   # Bottom wall
    return img


def test_process_returns_all_stages(test_floorplan_rgb):
    processor = ImageProcessor()
    stages = processor.process(test_floorplan_rgb)

    assert isinstance(stages, ProcessedStages)
    assert stages.original_rgb.shape == (100, 100, 3)
    assert stages.grayscale.shape == (100, 100)
    assert stages.denoised.shape == (100, 100)
    assert stages.threshold.shape == (100, 100)
    assert stages.edges.shape == (100, 100)
    assert stages.morphology.shape == (100, 100)
    assert stages.processing_time_ms >= 0.0
    assert "noise_filter" in stages.parameters


def test_source_image_immutability(test_floorplan_rgb):
    original_copy = np.copy(test_floorplan_rgb)
    processor = ImageProcessor()
    _ = processor.process(test_floorplan_rgb)

    # Verify original array was not modified in-place
    np.testing.assert_array_equal(test_floorplan_rgb, original_copy)


def test_thresholding_methods(test_floorplan_rgb):
    processor = ImageProcessor()

    # Test Gaussian adaptive
    stages_gauss = processor.process(test_floorplan_rgb, adaptive_method="gaussian")
    assert stages_gauss.threshold.ndim == 2

    # Test Mean adaptive
    stages_mean = processor.process(test_floorplan_rgb, adaptive_method="mean")
    assert stages_mean.threshold.ndim == 2

    # Test Otsu
    stages_otsu = processor.process(test_floorplan_rgb, adaptive_method="otsu")
    assert stages_otsu.threshold.ndim == 2


def test_bilateral_denoising(test_floorplan_rgb):
    processor = ImageProcessor()
    stages = processor.process(test_floorplan_rgb, use_bilateral=True)
    assert stages.parameters["noise_filter"] == "bilateral"
    assert stages.denoised.shape == (100, 100)


def test_edge_detection_output_binary(test_floorplan_rgb):
    processor = ImageProcessor()
    stages = processor.process(test_floorplan_rgb)

    # Canny edges should contain 0 or 255
    unique_vals = np.unique(stages.edges)
    for val in unique_vals:
        assert val in (0, 255)
