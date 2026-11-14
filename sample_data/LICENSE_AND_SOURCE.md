# Sample Architectural Floor Plan Fixture

## Overview
This file documents the sample test fixture provided in `sample_data/sample_floorplan.png`.

## Source and Provenance
- **Origin**: Programmatically generated using standard 2D vector drafting primitives in Python (`sample_data/generate_sample.py`).
- **License**: CC0 1.0 Universal (Public Domain Dedication). Free to use, modify, and distribute without restriction.
- **Resolution**: 800 x 600 pixels, RGB 8-bit PNG.
- **Architectural Layout**: A standard 2-bedroom residential apartment layout comprising:
  - Living / Dining Area
  - Kitchen
  - Hallway
  - Master Bedroom
  - Secondary Bedroom
  - Bathroom
  - Wall thickness: 8px (exterior walls), 6px (interior partitions)
  - Doorway gaps: ~50px standard openings

## Strict Computer Vision Protocol
- **No Hardcoding**: This test fixture is strictly processed through the identical OpenCV pipeline as any user-uploaded image (`ImageLoader` -> `ImageProcessor` -> `WallDetector` -> `RoomExtractor` -> `FloorPlanParser`).
- **No Prewritten Results**: No room boundaries, coordinates, or labels are hardcoded in the codebase for this or any other image.
- **Real-World Notice**: This synthetic image is provided as an offline test fixture for reproducible computer-vision testing. In real-world architectural drawings, scanned hand drawings, blueprints, or CAD exports with text annotations, detection quality will vary depending on scan resolution, line clarity, and boundary completeness.
