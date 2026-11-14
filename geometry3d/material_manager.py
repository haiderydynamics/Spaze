"""Architectural Material and Color Management System for SPAZE 3D Generation.

Provides curated architectural palettes, color mappings, and material attributes
for floors, interior/exterior walls, openings, glass, and procedural furniture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class Material:
    """Represents visual attributes for 3D surfaces."""
    name: str
    color_rgba: Tuple[float, float, float, float]  # Normalized 0.0 - 1.0
    hex_code: str
    roughness: float = 0.5
    metallic: float = 0.0
    double_sided: bool = True

    @property
    def color_rgb_uint8(self) -> Tuple[int, int, int]:
        """Return 8-bit integer RGB tuple."""
        return (
            int(self.color_rgba[0] * 255),
            int(self.color_rgba[1] * 255),
            int(self.color_rgba[2] * 255),
        )

    @property
    def color_rgba_uint8(self) -> Tuple[int, int, int, int]:
        """Return 8-bit integer RGBA tuple."""
        return (
            int(self.color_rgba[0] * 255),
            int(self.color_rgba[1] * 255),
            int(self.color_rgba[2] * 255),
            int(self.color_rgba[3] * 255),
        )


class MaterialManager:
    """Manages architectural colors and PBR materials across procedural geometries."""

    # Default Architectural Clean Palette
    PALETTES: Dict[str, Dict[str, Material]] = {
        "architectural_clean": {
            # Structural Materials
            "foundation": Material("Foundation", (0.85, 0.88, 0.90, 1.0), "#D9E0E6", roughness=0.8),
            "wall_exterior": Material("Exterior Wall", (0.91, 0.93, 0.95, 1.0), "#E8EDF2", roughness=0.85),
            "wall_interior": Material("Interior Wall", (0.96, 0.96, 0.97, 1.0), "#F5F5F7", roughness=0.9),
            "wall_lintel": Material("Wall Lintel", (0.88, 0.90, 0.92, 1.0), "#E0E5EB", roughness=0.85),
            
            # Openings & Glass
            "door_frame": Material("Door Frame", (0.45, 0.28, 0.16, 1.0), "#734729", roughness=0.4),
            "door_leaf": Material("Door Leaf", (0.62, 0.40, 0.24, 1.0), "#9E663D", roughness=0.45),
            "window_frame": Material("Window Frame", (0.35, 0.40, 0.46, 1.0), "#596675", roughness=0.3, metallic=0.4),
            "glass": Material("Window Glass", (0.55, 0.80, 0.95, 0.45), "#8CCCFF", roughness=0.1, metallic=0.1),

            # Room Floor Finishes
            "floor_living_room": Material("Living Room Oak", (0.89, 0.83, 0.75, 1.0), "#E3D4BF", roughness=0.4),
            "floor_bedroom": Material("Bedroom Soft Wood", (0.87, 0.84, 0.88, 1.0), "#DED6E0", roughness=0.6),
            "floor_kitchen": Material("Kitchen Ceramic Tile", (0.82, 0.92, 0.88, 1.0), "#D1EBE0", roughness=0.2),
            "floor_bathroom": Material("Bathroom Porcelain Tile", (0.80, 0.91, 0.95, 1.0), "#CCE8F2", roughness=0.15),
            "floor_balcony": Material("Balcony Slate", (0.75, 0.78, 0.82, 1.0), "#BFC7D1", roughness=0.7),
            "floor_corridor": Material("Corridor Stone", (0.92, 0.92, 0.94, 1.0), "#EBECEF", roughness=0.5),
            "floor_dining_room": Material("Dining Parquet", (0.91, 0.85, 0.76, 1.0), "#E8D9C2", roughness=0.45),
            "floor_default": Material("Default Floor", (0.90, 0.90, 0.92, 1.0), "#E6E6EB", roughness=0.5),

            # Furniture Materials
            "furniture_wood": Material("Furniture Wood", (0.58, 0.38, 0.22, 1.0), "#946138", roughness=0.5),
            "furniture_fabric_blue": Material("Fabric Soft Blue", (0.35, 0.55, 0.85, 1.0), "#598CD9", roughness=0.9),
            "furniture_fabric_gray": Material("Fabric Slate", (0.50, 0.55, 0.60, 1.0), "#808C99", roughness=0.9),
            "furniture_ceramic": Material("Sanitary Ceramic", (0.98, 0.98, 0.99, 1.0), "#FAFAFC", roughness=0.1),
            "furniture_metal": Material("Brushed Chrome", (0.75, 0.77, 0.80, 1.0), "#BFC4CC", roughness=0.25, metallic=0.7),
            "bed_sheet": Material("Bed Linen", (0.95, 0.95, 0.97, 1.0), "#F2F2F7", roughness=0.95),
            "bed_pillow": Material("Bed Pillow", (0.88, 0.88, 0.92, 1.0), "#E0E0EB", roughness=0.9),
        }
    }

    def __init__(self, palette_name: str = "architectural_clean") -> None:
        self.palette_name = palette_name if palette_name in self.PALETTES else "architectural_clean"
        self._palette = self.PALETTES[self.palette_name]

    def get_material(self, key: str) -> Material:
        """Fetch material by key with robust fallback."""
        if key in self._palette:
            return self._palette[key]
        return self._palette.get("floor_default", Material("Default", (0.8, 0.8, 0.8, 1.0), "#CCCCCC"))

    def get_floor_material_for_room(self, room_type: str) -> Material:
        """Resolve floor material matching room semantic type."""
        norm_type = (room_type or "").strip().lower().replace(" ", "_")
        key = f"floor_{norm_type}"
        if key in self._palette:
            return self._palette[key]
        return self.get_material("floor_default")

    def get_rgba_uint8(self, key: str) -> Tuple[int, int, int, int]:
        """Convenience method returning 8-bit RGBA tuple."""
        return self.get_material(key).color_rgba_uint8

    def get_rgb_uint8(self, key: str) -> Tuple[int, int, int]:
        """Convenience method returning 8-bit RGB tuple."""
        return self.get_material(key).color_rgb_uint8
