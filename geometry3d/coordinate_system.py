"""Coordinate System Transformation Engine for SPAZE 3D Generation.

Provides mathematically rigorous, reversible mappings between:
1. 2D Computer Vision Floor-Plan Coordinates (Top-Left origin, Y-down, pixels)
2. 3D World Cartesian Coordinates (Z-up elevation, meters, right-handed system)
Guarantees zero unintended horizontal or vertical mirroring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import Polygon

from geometry3d.config import Geometry3DConfig
from models.floorplan_schema import FloorPlan, ScaleInfo


@dataclass(frozen=True)
class CoordinateMetadata:
    """Stores coordinate system configuration and provenance."""
    is_calibrated: bool
    scale_meters_per_pixel: float
    image_width: int
    image_height: int
    building_width_m: float
    building_depth_m: float
    coordinate_convention: str = "right_handed_z_up"


class CoordinateTransformer:
    """Transforms 2D floor-plan geometry into 3D world coordinates."""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        scale_info: Optional[ScaleInfo] = None,
        config: Optional[Geometry3DConfig] = None,
    ) -> None:
        self.image_width = max(1, image_width)
        self.image_height = max(1, image_height)
        self.config = config or Geometry3DConfig()

        # Determine scale factor (meters per pixel)
        if scale_info is not None and scale_info.is_calibrated and scale_info.pixels_per_meter and scale_info.pixels_per_meter > 0:
            self.scale_m_per_px = 1.0 / float(scale_info.pixels_per_meter)
            self.is_calibrated = True
        else:
            self.scale_m_per_px = 1.0 / float(self.config.default_pixels_per_meter)
            self.is_calibrated = False

        self.building_width_m = round(self.image_width * self.scale_m_per_px, 3)
        self.building_depth_m = round(self.image_height * self.scale_m_per_px, 3)

    @classmethod
    def from_floorplan(
        cls,
        floorplan: FloorPlan,
        config: Optional[Geometry3DConfig] = None,
    ) -> CoordinateTransformer:
        """Construct transformer directly from a FloorPlan schema instance."""
        return cls(
            image_width=floorplan.image_width,
            image_height=floorplan.image_height,
            scale_info=floorplan.scale_information,
            config=config,
        )

    def to_3d_point(self, x_2d: float, y_2d: float, z_elevation_m: float = 0.0) -> Tuple[float, float, float]:
        """Convert a 2D pixel coordinate (top-left origin) to 3D world meters (Z-up).

        Formulation:
            X_3d = x_2d * scale
            Y_3d = (H_image - y_2d) * scale   (inverts Y-down into Cartesian Y-up)
            Z_3d = z_elevation_m
        """
        x_3d = float(x_2d) * self.scale_m_per_px
        y_3d = float(self.image_height - y_2d) * self.scale_m_per_px
        return (round(x_3d, 4), round(y_3d, 4), round(float(z_elevation_m), 4))

    def to_2d_point(self, x_3d: float, y_3d: float) -> Tuple[float, float]:
        """Reversibly map a 3D world coordinate back to 2D image pixels."""
        x_2d = x_3d / self.scale_m_per_px
        y_2d = self.image_height - (y_3d / self.scale_m_per_px)
        return (round(x_2d, 2), round(y_2d, 2))

    def to_3d_polygon(
        self,
        polygon_2d: List[Tuple[float, float]],
        z_elevation_m: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """Transform 2D polygon vertices into 3D world points."""
        return [self.to_3d_point(pt[0], pt[1], z_elevation_m) for pt in polygon_2d]

    def to_3d_planar_polygon(
        self,
        polygon_2d: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Transform 2D pixel polygon into 2D world-meter polygon on the XY plane."""
        pts_3d = [self.to_3d_point(pt[0], pt[1], 0.0) for pt in polygon_2d]
        return [(p[0], p[1]) for p in pts_3d]

    def to_meters(self, pixel_distance: float) -> float:
        """Convert a 1D pixel length or thickness to real-world meters."""
        return float(pixel_distance) * self.scale_m_per_px

    def to_pixels(self, meter_distance: float) -> float:
        """Convert a meter distance back to pixels."""
        return float(meter_distance) / self.scale_m_per_px

    def get_metadata(self) -> CoordinateMetadata:
        """Retrieve coordinate configuration metadata."""
        return CoordinateMetadata(
            is_calibrated=self.is_calibrated,
            scale_meters_per_pixel=self.scale_m_per_px,
            image_width=self.image_width,
            image_height=self.image_height,
            building_width_m=self.building_width_m,
            building_depth_m=self.building_depth_m,
            coordinate_convention="right_handed_z_up",
        )
