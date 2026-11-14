"""Visualization Module for SPAZE Generated Floor Plans.

Provides clear 2D architectural visualizations comparing source floor plans
against generated alternatives using Matplotlib.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from models.floorplan_schema import FloorPlan, Room


# Architectural color palette for standard room types
ROOM_COLOR_MAP: Dict[str, Tuple[float, float, float, float]] = {
    "Living Room": (0.20, 0.60, 0.86, 0.65),     # Electric Sky Blue
    "Master Bedroom": (0.60, 0.35, 0.71, 0.65),  # Deep Amethyst Purple
    "Bedroom": (0.55, 0.45, 0.75, 0.60),         # Lavender
    "Kitchen": (0.95, 0.61, 0.07, 0.65),         # Golden Amber
    "Dining Room": (0.90, 0.49, 0.13, 0.65),     # Burnt Orange
    "Bathroom": (0.10, 0.74, 0.61, 0.65),        # Mint Teal
    "Hallway": (0.58, 0.65, 0.65, 0.50),         # Neutral Slate Grey
    "Study Room": (0.18, 0.80, 0.44, 0.65),      # Emerald Green
    "Balcony": (0.24, 0.70, 0.44, 0.55),         # Seafoam
    "Utility Room": (0.50, 0.55, 0.55, 0.50),    # Steel
    "Other": (0.45, 0.50, 0.55, 0.50),
    "Unclassified": (0.50, 0.50, 0.50, 0.40),
}


def render_single_floorplan_axes(
    ax: plt.Axes,
    floorplan: FloorPlan,
    title: str = "Floor Plan",
    show_labels: bool = True,
) -> None:
    """Draw a single floor plan onto a Matplotlib Axes."""
    w = floorplan.image_width
    h = floorplan.image_height

    # Dark architectural canvas background
    ax.set_facecolor("#0f172a")
    ax.set_xlim(-20, w + 20)
    ax.set_ylim(h + 20, -20)  # Invert Y so origin (0,0) is top-left
    ax.set_aspect("equal")

    # Building boundary wall
    boundary = patches.Rectangle(
        (0, 0), w, h,
        linewidth=2.5,
        edgecolor="#38bdf8",
        facecolor="none",
        linestyle="-",
    )
    ax.add_patch(boundary)

    rooms = floorplan.confirmed_rooms + floorplan.candidate_rooms
    scale = floorplan.scale_information

    for r in rooms:
        color = ROOM_COLOR_MAP.get(r.room_type, ROOM_COLOR_MAP["Other"])

        # Room rectangle
        rect = patches.Rectangle(
            (r.x, r.y), r.width, r.height,
            linewidth=1.8,
            edgecolor=(color[0], color[1], color[2], 1.0),
            facecolor=color,
        )
        ax.add_patch(rect)

        if show_labels:
            cx = r.x + r.width / 2.0
            cy = r.y + r.height / 2.0

            if scale.is_calibrated and r.physical_width and r.physical_height:
                dim_str = f"{r.physical_width:.1f}m x {r.physical_height:.1f}m"
            else:
                dim_str = f"{int(r.width)}x{int(r.height)}px"

            ax.text(
                cx, cy - 6,
                r.label or r.room_type,
                color="#ffffff",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
            )
            ax.text(
                cx, cy + 8,
                dim_str,
                color="#cbd5e1",
                fontsize=6.5,
                ha="center",
                va="center",
            )

    ax.set_title(title, color="#e2e8f0", fontsize=9.5, fontweight="bold", pad=8)
    ax.tick_params(colors="#64748b", labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def visualize_alternatives_comparison(
    source_floorplan: FloorPlan,
    alternatives: List[FloorPlan],
    preference: str = "space_efficient",
    diversity_score: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create a side-by-side 4-panel comparison figure and optionally save to disk."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), facecolor="#0b0f17")

    # Panel 0: Source Floor Plan
    src_title = f"Source Floor Plan\n({len(source_floorplan.confirmed_rooms + source_floorplan.candidate_rooms)} Rooms)"
    render_single_floorplan_axes(axes[0], source_floorplan, title=src_title)

    # Panels 1-3: Alternatives
    pref_label = preference.replace("_", " ").title()
    for idx in range(3):
        ax = axes[idx + 1]
        if idx < len(alternatives):
            alt = alternatives[idx]
            alt_rooms = len(alt.confirmed_rooms + alt.candidate_rooms)
            alt_title = f"Alternative {idx + 1} ({pref_label})\n({alt_rooms} Rooms)"
            render_single_floorplan_axes(ax, alt, title=alt_title)
        else:
            ax.set_facecolor("#0b0f17")
            ax.text(0.5, 0.5, "No Candidate", color="#64748b", ha="center", va="center")
            ax.axis("off")

    fig.suptitle(
        f"SPAZE Generative AI Engine — Layout Alternatives (Preference: {pref_label} | Diversity: {diversity_score:.2f})",
        color="#38bdf8",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_p), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"Comparison visualization saved to {out_p}")

    return fig
