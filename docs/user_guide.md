# SPAZE — Comprehensive User Guide

## 1. Navigating the Unified Workspace
SPAZE is organized into 7 sequential workflow stages accessible via the persistent left sidebar:

1. **00 — Workspace Overview**: Project dashboard, statistics, active session state, and quick-start actions.
2. **01 — Floor Plan Studio**: 2D blueprint upload, local OpenCV extraction, scale calibration, and room verification.
3. **02 — AI Layout Generator**: Preference-conditioned CVAE layout synthesis (3 distinct alternatives).
4. **03 — 3D Visualization Studio**: Interactive WebGL orbit viewer, physical openings, floor slabs, and furniture.
5. **04 — Evaluation Lab**: Ground-truth CV metrics, CVAE loss telemetry, and procedural baseline benchmarks.
6. **05 — Project Export Hub**: One-click multi-format deliverables (GLB, JSON Schema, 3D Manifest, Blueprint, CSV).
7. **06 — Academic Research & Docs**: Interactive reader for all technical specifications and research chapters.

---

## 2. Stage-by-Stage Operating Instructions

### Stage 01: Ingesting & Verifying a Floor Plan
1. **Upload or Sample**:
   - Click **"📂 Load Sample Test Fixture"** to immediately test the bundled 800×600 residential drawing, or drag and drop your own 2D floor plan (`.png`, `.jpg`, `.jpeg`).
2. **Review CV Extraction**:
   - Switch through intermediate inspection tabs: *Annotated Rooms*, *Wall Candidates*, *Threshold*, *Canny Edges*, *Morphology*.
3. **Calibrate Physical Scale**:
   - In the **Physical Scale Calibration** expander, enter a known pixel distance (e.g., $680\,\text{px}$) and real-world length in meters (e.g., $13.6\,\text{m}$), then click **Apply Scale**. All dimensions update dynamically to square meters.
4. **Human-in-the-Loop Room Editor**:
   - Select any detected room to rename labels, reassign types (e.g., Living Room $\to$ Master Bedroom), adjust pixel coordinates, or click **✅ Confirm** / **❌ Reject**.
   - Use **"➕ Add Room Manually"** to draw any missed alcove or balcony.
5. Click **"👉 Continue to AI Generation (Stage 02)"**.

---

### Stage 02: Synthesizing Layout Alternatives
1. **Select Architectural Preference**:
   - Choose one of three design conditioning vectors:
     - 📐 **Space-Efficient**: Minimal circulation, compact room clusters.
     - 🛋️ **Open Layout**: Connected common spaces, shared boundary borders.
     - 🚪 **Privacy-Focused**: Isolated bedroom clusters, corridor buffering.
2. **Adjust Sampling Controls (Optional)**:
   - Expand *Tuning Parameters* to adjust Latent Temperature $\tau \in [0.5, 2.0]$ (higher values yield greater structural variation) and Random Seed for reproducible sampling.
3. Click **"🚀 Generate 3 Alternative Layouts"**.
4. **Inspect Alternatives**:
   - Compare the Source floor plan against Alternative 1, Alternative 2, and Alternative 3 side-by-side.
   - Review Diversity Score ($1 - \text{IoU}$) and Geometric Validity Rate ($100\%$).
5. **Select Preferred Layout**:
   - Choose the alternative you want to inhabit in 3D using the selection dropdown.
6. Click **"👉 Proceed to 3D Visualization (Stage 03)"**.

---

### Stage 03: Exploring & Customizing the 3D Model
1. **Interactive WebGL Controls**:
   - **Orbit**: Left-click and drag across the 3D model.
   - **Zoom**: Scroll wheel up/down.
   - **Pan**: Right-click and drag.
   - **Layer Toggles**: Click items in the legend or check/uncheck semantic controls on the right panel.
2. **Adjust Model Parameters**:
   - Real-time sliders for Wall Height ($2.2\,\text{m} - 4.2\,\text{m}$), Exterior Wall Thickness ($0.15\,\text{m} - 0.40\,\text{m}$), and Interior Wall Thickness ($0.08\,\text{m} - 0.20\,\text{m}$).
   - Toggle visibility of Doors, Windows, Floors, and Procedural Furniture.
   - Click **"🔨 Recompute 3D Geometry"** to regenerate the mesh instantly.
3. **Download Deliverables**:
   - Download the 3D model directly as `.glb`, manifest as `.json`, or blueprint as `.png`.
4. Click **"👉 Stage 04: Evaluation & Export"**.

---

### Stage 04 & 05: Evaluation Lab & Project Export Hub
1. **Inspect Research Metrics**:
   - Review Computer Vision precision/recall against CAD ground truth.
   - Analyze CVAE training convergence curves and comparative baseline benchmark tables.
2. **Export Complete Bundle**:
   - In Stage 05 (Project Export Hub), download your complete design deliverables: GLB 3D Model, Floor Plan JSON Schema, 3D Building Manifest, 2D Blueprint Image, and Research Benchmark CSV.
