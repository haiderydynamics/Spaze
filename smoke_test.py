"""SPAZE Dependency and Environment Smoke-Test Script.

Verifies the integrity of all core scientific, deep learning, geometric,
and visualization libraries, ensuring clean execution in any Python environment.
"""
from __future__ import annotations

import sys
from pathlib import Path


def run_smoke_test() -> bool:
    print("=" * 60)
    print(" SPAZE ENVIRONMENT & DEPENDENCY SMOKE TEST")
    print("=" * 60)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split()[0]}")

    all_ok = True

    # 1. PyTorch & Torchvision
    try:
        import torch
        import torchvision
        dev = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        print(f" [PASS] PyTorch {torch.__version__} ({dev})")
        print(f" [PASS] Torchvision {torchvision.__version__}")
    except Exception as e:
        print(f" [FAIL] PyTorch / Torchvision: {e}")
        all_ok = False

    # 2. Computer Vision & Imaging
    try:
        import cv2
        import numpy as np
        import PIL
        import matplotlib
        print(f" [PASS] OpenCV {cv2.__version__}")
        print(f" [PASS] NumPy {np.__version__}")
        print(f" [PASS] Pillow {PIL.__version__}")
        print(f" [PASS] Matplotlib {matplotlib.__version__}")
    except Exception as e:
        print(f" [FAIL] CV / Imaging: {e}")
        all_ok = False

    # 3. Computational Geometry & 3D
    try:
        import shapely
        import trimesh
        import mapbox_earcut
        import scipy
        print(f" [PASS] Shapely {shapely.__version__}")
        print(f" [PASS] Trimesh {trimesh.__version__}")
        print(f" [PASS] Mapbox-Earcut {mapbox_earcut.__version__}")
        print(f" [PASS] SciPy {scipy.__version__}")
    except Exception as e:
        print(f" [FAIL] Geometry / 3D: {e}")
        all_ok = False

    # 4. Web & Interactive UI
    try:
        import plotly
        import pandas as pd
        import streamlit
        print(f" [PASS] Plotly {plotly.__version__}")
        print(f" [PASS] Pandas {pd.__version__}")
        print(f" [PASS] Streamlit {streamlit.__version__}")
    except Exception as e:
        print(f" [FAIL] UI / Visualization: {e}")
        all_ok = False

    # 5. Checkpoint Verification
    ckpt_path = Path("checkpoints/spaze_cvae_best.pth")
    if ckpt_path.is_file():
        size_kb = ckpt_path.stat().st_size / 1024.0
        print(f" [PASS] CVAE Checkpoint verified ({size_kb:.1f} KB at {ckpt_path})")
    else:
        print(f" [WARN] CVAE Checkpoint missing at {ckpt_path}")

    # 6. Procedural 3D Engine Verification
    try:
        import geometry3d
        from models.floorplan_schema import FloorPlan, Room
        fp = FloorPlan(
            image_width=800,
            image_height=600,
            confirmed_rooms=[
                Room(room_id="r1", room_type="living_room", label="Living", x=100, y=100, width=300, height=300),
            ]
        )
        res = geometry3d.generate_3d_model(fp)
        glb = res.get_glb_bytes()
        valid_glb = geometry3d.ModelExporter.verify_glb(glb)
        print(f" [PASS] Procedural 3D Engine: Synthesized {res.metrics.total_triangles} triangles (Valid GLB: {valid_glb})")
    except Exception as e:
        print(f" [FAIL] Procedural 3D Engine: {e}")
        all_ok = False

    print("=" * 60)
    if all_ok:
        print(" ALL DEPENDENCY CHECKS PASSED SUCCESSFULLY!")
    else:
        print(" SMOKE TEST DETECTED ISSUES")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
