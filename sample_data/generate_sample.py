"""Script to generate a standardized architectural test fixture floor plan.

Draws a clean 2D architectural layout (Living Room, Kitchen, Bedroom 1, Bedroom 2, Bathroom, Hallway)
with realistic wall thicknesses, door openings, and room boundaries.
This image serves as a test fixture for the local computer vision pipeline.
"""
from pathlib import Path
import cv2
import numpy as np


def generate_architectural_floorplan(output_path: Path) -> None:
    # 800 x 600 canvas, white background
    width, height = 800, 600
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Wall styling
    wall_color = (30, 30, 30)  # Dark charcoal/black
    ext_thick = 8
    int_thick = 6

    # Exterior Boundary (margin: 60px)
    x0, y0 = 60, 60
    x1, y1 = 740, 540

    # Draw exterior walls
    cv2.line(img, (x0, y0), (x1, y0), wall_color, ext_thick)
    cv2.line(img, (x1, y0), (x1, y1), wall_color, ext_thick)
    cv2.line(img, (x1, y1), (x0, y1), wall_color, ext_thick)
    cv2.line(img, (x0, y1), (x0, y0), wall_color, ext_thick)

    # Interior layout coordinates:
    # Split horizontally at y = 280
    # Living Room (left: x0 to 420, y0 to 340)
    # Kitchen (bottom-left: x0 to 280, 340 to y1)
    # Hallway (280 to 420, 340 to y1)
    # Bedroom 1 (top-right: 420 to x1, y0 to 300)
    # Bedroom 2 (bottom-right: 420 to x1, 380 to y1)
    # Bathroom (middle-right: 420 to x1, 300 to 380)

    # Horizontal divider at y=340 from x0 to 420 (Living/Kitchen) with a door gap
    # Wall from 60 to 220, gap 220-270, wall 270-420
    cv2.line(img, (60, 340), (220, 340), wall_color, int_thick)
    cv2.line(img, (270, 340), (420, 340), wall_color, int_thick)

    # Vertical divider at x=420 from y0 (60) to y1 (540) with door openings
    # Wall from 60 to 180, gap 180-230, wall 230-340, gap 340-390, wall 390-540
    cv2.line(img, (420, 60), (420, 180), wall_color, int_thick)
    cv2.line(img, (420, 230), (420, 340), wall_color, int_thick)
    cv2.line(img, (420, 390), (420, 540), wall_color, int_thick)

    # Vertical divider at x=280 from 340 to y1 (Kitchen/Hallway) with door opening
    # Wall from 340 to 440, gap 440-490, wall 490 to 540
    cv2.line(img, (280, 340), (280, 440), wall_color, int_thick)
    cv2.line(img, (280, 490), (280, 540), wall_color, int_thick)

    # Horizontal divider at y=300 from 420 to 740 (Bedroom 1 / Bathroom) with door opening
    # Wall from 420 to 520, gap 520-570, wall 570 to 740
    cv2.line(img, (420, 300), (520, 300), wall_color, int_thick)
    cv2.line(img, (570, 300), (740, 300), wall_color, int_thick)

    # Horizontal divider at y=380 from 420 to 740 (Bathroom / Bedroom 2) with door opening
    # Wall from 420 to 520, gap 520-570, wall 570 to 740
    cv2.line(img, (420, 380), (520, 380), wall_color, int_thick)
    cv2.line(img, (570, 380), (740, 380), wall_color, int_thick)

    # Add realistic architectural dimension lines and room annotations
    # Subtle dimension line on top
    cv2.line(img, (60, 40), (740, 40), (120, 120, 120), 1)
    cv2.line(img, (60, 35), (60, 45), (120, 120, 120), 1)
    cv2.line(img, (740, 35), (740, 45), (120, 120, 120), 1)
    cv2.putText(img, "13.6 m", (370, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # Save to path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Generated sample floor plan at {output_path} ({width}x{height})")


if __name__ == "__main__":
    out = Path("sample_data/sample_floorplan.png")
    generate_architectural_floorplan(out)
