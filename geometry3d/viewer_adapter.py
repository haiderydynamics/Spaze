"""Plotly 3D Web Viewer Adapter for Interactive Architectural Models.

Converts trimesh.Scene graphs into rich, high-performance Plotly 3D Mesh traces
supporting orbit, zoom, pan, granular semantic layer toggling, and architectural camera presets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import trimesh


class ViewerAdapter:
    """Adapts Trimesh 3D scenes into interactive Plotly figures."""

    @staticmethod
    def scene_to_plotly(
        scene: trimesh.Scene,
        camera_mode: str = "isometric",
        theme: str = "light",
        show_floors: bool = True,
        show_exterior_walls: bool = True,
        show_interior_walls: bool = True,
        show_doors: bool = True,
        show_windows: bool = True,
        show_furniture: bool = True,
    ) -> go.Figure:
        """Convert a trimesh.Scene to a Plotly go.Figure with fine-grained layer visibility.

        Args:
            scene: Populated trimesh.Scene instance.
            camera_mode: One of 'isometric', 'perspective', 'top_down'.
            theme: 'dark' or 'light'.
            show_floors: Toggle floor slabs and foundation.
            show_exterior_walls: Toggle exterior envelope walls.
            show_interior_walls: Toggle interior partition walls.
            show_doors: Toggle door frames and entrance accents.
            show_windows: Toggle window glass and trim.
            show_furniture: Toggle procedural furniture.

        Returns:
            Configured plotly.graph_objects.Figure.
        """
        fig = go.Figure()

        # Group geometries into granular architectural layers
        layers: Dict[str, List[Tuple[str, trimesh.Trimesh]]] = {
            "Floors": [],
            "Exterior Walls": [],
            "Interior Walls": [],
            "Doors": [],
            "Windows": [],
            "Furniture": [],
        }

        for node_name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
                continue

            category = str(geom.metadata.get("semantic_category") or geom.metadata.get("semantic_node") or "").lower()
            wall_type = str(geom.metadata.get("wall_type") or "").lower()

            if "foundation" in category or "floor" in category:
                if show_floors:
                    layers["Floors"].append((node_name, geom))
            elif "exterior" in category or wall_type == "exterior":
                if show_exterior_walls:
                    layers["Exterior Walls"].append((node_name, geom))
            elif "interior" in category or wall_type == "interior":
                if show_interior_walls:
                    layers["Interior Walls"].append((node_name, geom))
            elif "door" in category:
                if show_doors:
                    layers["Doors"].append((node_name, geom))
            elif "window" in category or "glass" in category:
                if show_windows:
                    layers["Windows"].append((node_name, geom))
            elif "furniture" in category:
                if show_furniture:
                    layers["Furniture"].append((node_name, geom))
            else:
                # Default fallback
                if show_exterior_walls:
                    layers["Exterior Walls"].append((node_name, geom))

        layer_default_colors = {
            "Floors": "#E2E8F0",
            "Exterior Walls": "#CBD5E1",
            "Interior Walls": "#E2E8F0",
            "Doors": "#92400E",
            "Windows": "#38BDF8",
            "Furniture": "#818CF8",
        }

        # Add traces
        for layer_name, geom_list in layers.items():
            if not geom_list:
                continue

            for node_name, mesh in geom_list:
                verts = mesh.vertices
                faces = mesh.faces

                # Build contextual hover tooltip
                hover_info = f"<b>{layer_name}</b>"
                if "room_name" in mesh.metadata:
                    hover_info += f"<br>Room: {mesh.metadata['room_name']}"
                if "room_type" in mesh.metadata:
                    hover_info += f"<br>Type: {mesh.metadata['room_type'].replace('_', ' ').title()}"
                if "area_sqm" in mesh.metadata:
                    hover_info += f"<br>Area: {mesh.metadata['area_sqm']} m²"
                if "wall_type" in mesh.metadata:
                    hover_info += f"<br>Partition: {mesh.metadata['wall_type'].title()}"

                # Vertex colors if present
                hex_color = layer_default_colors.get(layer_name, "#A0AEC0")
                if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
                    try:
                        c = mesh.visual.vertex_colors[0]
                        hex_color = f"rgba({c[0]},{c[1]},{c[2]},{c[3]/255.0:.2f})"
                    except Exception:
                        pass

                trace = go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=hex_color,
                    opacity=0.45 if layer_name == "Windows" else 1.0,
                    name=mesh.metadata.get("room_name", layer_name),
                    legendgroup=layer_name,
                    showlegend=(layer_name in ["Floors", "Exterior Walls", "Interior Walls", "Doors", "Windows", "Furniture"]),
                    hoverinfo="text",
                    text=hover_info,
                    lighting=dict(
                        ambient=0.65,
                        diffuse=0.85,
                        fresnel=0.2,
                        specular=0.25,
                        roughness=0.5,
                    ),
                )
                fig.add_trace(trace)

        # Normalize camera mode key
        norm_cam = camera_mode.lower().replace(" ", "_")
        cameras = {
            "isometric": dict(
                eye=dict(x=1.5, y=-1.5, z=1.3),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            "isometric_view": dict(
                eye=dict(x=1.5, y=-1.5, z=1.3),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            "perspective": dict(
                eye=dict(x=1.8, y=-1.8, z=0.9),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            "perspective_view": dict(
                eye=dict(x=1.8, y=-1.8, z=0.9),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            "top_down": dict(
                eye=dict(x=0.001, y=0.001, z=2.6),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
            ),
            "top-down_3d_view": dict(
                eye=dict(x=0.001, y=0.001, z=2.6),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
            ),
        }

        camera = cameras.get(norm_cam, cameras["isometric"])
        bg_color = "#0F172A" if theme == "dark" else "#F8FAFC"
        grid_color = "#334155" if theme == "dark" else "#E2E8F0"
        text_color = "#E2E8F0" if theme == "dark" else "#334155"

        fig.update_layout(
            template="plotly_white" if theme == "light" else "plotly_dark",
            margin=dict(l=0, r=0, b=0, t=30),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            scene=dict(
                camera=camera,
                aspectmode="data",
                xaxis=dict(
                    title=dict(text="Width (X m)", font=dict(color=text_color, size=11)),
                    showbackground=True,
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    tickfont=dict(color=text_color, size=9),
                ),
                yaxis=dict(
                    title=dict(text="Depth (Y m)", font=dict(color=text_color, size=11)),
                    showbackground=True,
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    tickfont=dict(color=text_color, size=9),
                ),
                zaxis=dict(
                    title=dict(text="Height (Z m)", font=dict(color=text_color, size=11)),
                    showbackground=True,
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    zerolinecolor=grid_color,
                    tickfont=dict(color=text_color, size=9),
                ),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5,
                font=dict(color=text_color, size=10),
            ),
        )

        return fig
