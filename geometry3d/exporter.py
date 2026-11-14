"""Multi-Format Architectural 3D Export Engine.

Exports procedural 3D models to industry-standard binary GLTF/GLB (.glb),
detailed JSON metadata manifests, and 2D architectural preview blueprints.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from geometry3d.config import Geometry3DConfig
from models.floorplan_schema import FloorPlan


class ModelExporter:
    """Manages exporting 3D models and accompanying architectural deliverables."""

    @staticmethod
    def export_glb(
        scene: trimesh.Scene,
        file_path_or_buffer: Optional[Union[str, io.BytesIO]] = None,
    ) -> bytes:
        """Export trimesh.Scene to standard binary GLTF 2.0 (.glb) format.

        Args:
            scene: Scene to export.
            file_path_or_buffer: Optional file path or BytesIO buffer.

        Returns:
            Raw bytes of the .glb file.
        """
        glb_bytes = scene.export(file_type="glb")
        if isinstance(glb_bytes, str):
            glb_bytes = glb_bytes.encode("utf-8")

        if file_path_or_buffer is not None:
            if isinstance(file_path_or_buffer, str):
                with open(file_path_or_buffer, "wb") as f:
                    f.write(glb_bytes)
            elif isinstance(file_path_or_buffer, io.BytesIO):
                file_path_or_buffer.write(glb_bytes)
                file_path_or_buffer.seek(0)

        return glb_bytes

    @staticmethod
    def verify_glb(glb_data: Union[bytes, str]) -> bool:
        """Verify that a generated GLB file is structurally valid and can be reloaded."""
        try:
            if isinstance(glb_data, bytes):
                buf = io.BytesIO(glb_data)
                loaded = trimesh.load(buf, file_type="glb")
            else:
                loaded = trimesh.load(glb_data, file_type="glb")

            # Check that it contains geometries
            if isinstance(loaded, trimesh.Scene):
                return len(loaded.geometry) > 0
            elif isinstance(loaded, trimesh.Trimesh):
                return len(loaded.vertices) > 0
            return False
        except Exception:
            return False

    @staticmethod
    def export_manifest(
        scene: trimesh.Scene,
        floorplan: FloorPlan,
        config: Geometry3DConfig,
        file_path_or_buffer: Optional[Union[str, io.StringIO]] = None,
    ) -> Dict[str, Any]:
        """Export comprehensive architectural JSON metadata manifest."""
        total_tris = 0
        total_verts = 0
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                total_tris += len(geom.faces)
                total_verts += len(geom.vertices)

        rooms = floorplan.confirmed_rooms + floorplan.candidate_rooms
        room_manifest = [
            {
                "id": r.id,
                "label": r.label,
                "room_type": r.room_type,
                "area_sqm": round(r.area_sqm, 2) if r.area_sqm else None,
                "width_m": round(r.width_m, 2) if r.width_m else None,
                "height_m": round(r.height_m, 2) if r.height_m else None,
            }
            for r in rooms
        ]

        bounds = scene.bounds
        extents = scene.extents if bounds is not None else [0.0, 0.0, 0.0]

        manifest = {
            "format": "SPAZE_3D_MANIFEST_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_image": floorplan.image_path,
            "canvas_resolution": [floorplan.image_width, floorplan.image_height],
            "units": "meters",
            "coordinate_system": "Right-Handed, Z-Up Elevation",
            "building_bounding_box_meters": {
                "width_x": round(float(extents[0]), 3),
                "depth_y": round(float(extents[1]), 3),
                "height_z": round(float(extents[2]), 3),
            },
            "statistics": {
                "total_triangles": total_tris,
                "total_vertices": total_verts,
                "mesh_node_count": len(scene.geometry),
                "room_count": len(rooms),
            },
            "configuration": {
                "wall_height_m": config.default_wall_height,
                "exterior_wall_thickness_m": config.exterior_wall_thickness,
                "interior_wall_thickness_m": config.interior_wall_thickness,
                "floor_thickness_m": config.floor_thickness,
                "palette": config.palette_name,
                "has_furniture": config.generate_furniture,
            },
            "rooms": room_manifest,
        }

        if file_path_or_buffer is not None:
            json_str = json.dumps(manifest, indent=2)
            if isinstance(file_path_or_buffer, str):
                with open(file_path_or_buffer, "w", encoding="utf-8") as f:
                    f.write(json_str)
            elif isinstance(file_path_or_buffer, io.StringIO):
                file_path_or_buffer.write(json_str)
                file_path_or_buffer.seek(0)

        return manifest

    @staticmethod
    def export_floorplan_preview(
        floorplan: FloorPlan,
        file_path_or_buffer: Optional[Union[str, io.BytesIO]] = None,
    ) -> bytes:
        """Render a clean 2D architectural blueprint preview of the floor plan."""
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.set_facecolor("#1E293B")
        fig.patch.set_facecolor("#0F172A")

        rooms = floorplan.confirmed_rooms + floorplan.candidate_rooms
        colors = ["#38BDF8", "#818CF8", "#34D399", "#FBBF24", "#F472B6", "#A78BFA", "#4ADE80"]

        for idx, r in enumerate(rooms):
            if len(r.polygon) >= 3:
                poly_pts = np.array(r.polygon)
                col = colors[idx % len(colors)]
                ax.fill(poly_pts[:, 0], poly_pts[:, 1], color=col, alpha=0.35)
                ax.plot(
                    np.append(poly_pts[:, 0], poly_pts[0, 0]),
                    np.append(poly_pts[:, 1], poly_pts[0, 1]),
                    color=col,
                    linewidth=2.0,
                )
                # Centroid label
                cx = float(np.mean(poly_pts[:, 0]))
                cy = float(np.mean(poly_pts[:, 1]))
                label = f"{r.label or r.room_type}\n{r.area_sqm:.1f}m²" if r.area_sqm else (r.label or r.room_type)
                ax.text(
                    cx, cy, label,
                    color="#FFFFFF",
                    fontsize=9,
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", alpha=0.8, edgecolor=col),
                )

        ax.set_xlim(0, max(1, floorplan.image_width))
        ax.set_ylim(max(1, floorplan.image_height), 0)  # Invert for CV image coordinates
        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        img_bytes = buf.getvalue()

        if file_path_or_buffer is not None:
            if isinstance(file_path_or_buffer, str):
                with open(file_path_or_buffer, "wb") as f:
                    f.write(img_bytes)
            elif isinstance(file_path_or_buffer, io.BytesIO):
                file_path_or_buffer.write(img_bytes)
                file_path_or_buffer.seek(0)

        return img_bytes
