<<<<<<< HEAD
# SPAZE — Phase 1: Foundation, Floor-Plan Image Processing & Structured Room Extraction

> **Tagline:** Imagine. Generate. Inhabit.  
> **Phase:** Phase 1 (Foundation & Floor-Plan Processing)  
> **Status:** Completed & Verified  

---

## 1. Project Overview

**SPAZE** is an AI-assisted architectural layout generation application. Its long-term vision is to accept a 2D architectural floor-plan image, parse its spatial structure, generate optimized room layout alternatives using a Conditional Variational Autoencoder (CVAE) in Phase 2, validate architectural constraints in Phase 3, and convert validated layouts into interactive, navigable 3D environments.

### Phase 1 Scope & Boundary
Phase 1 delivers the architectural foundation:
- Robust local decoding and verification of PNG, JPG, and JPEG floor-plan images.
- A deterministic, 5-stage OpenCV computer vision pipeline (Grayscale -> Bilateral/Gaussian Denoising -> Adaptive/Otsu Thresholding -> Canny Edges -> Morphological Wall Enhancement).
- Wall-line candidate detection with architectural orientation heuristics (horizontal, vertical, diagonal).
- Candidate room region extraction and polygon approximation with transparent, explainable heuristic scoring.
- Physical scale calibration establishing real-world meters from pixel distances.
- Human-in-the-loop room editing (correcting labels, types, bounds, status, and manual room creation).
- Standardized, decoupled dataclass schema (`FloorPlan`, `Room`, `WallCandidate`, `ScaleInfo`) and formatted JSON export/import.
- Modern dark-charcoal Streamlit interface.

> [!IMPORTANT]
> Phase 1 deliberately does not implement the CVAE, generative AI layout synthesis, or 3D rendering (reserved for Phase 2 and Phase 3). However, the structured schema is designed specifically as the contract for those downstream engines.

---

## 2. Project Folder Structure

```
SPAZE/
├── app.py                     # Streamlit application UI and interaction flow
├── requirements.txt           # Project dependencies
├── README.md                  # Complete technical documentation
├── .gitignore                 # Version control exclusions
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Central constants, limits, thresholds, and room types
│
├── models/
│   ├── __init__.py
│   └── floorplan_schema.py    # Dataclasses: FloorPlan, Room, WallCandidate, ScaleInfo
│
├── modules/
│   ├── __init__.py
│   ├── image_loader.py        # Magic-byte validation, format decoding, dimension limits
│   ├── image_processor.py     # Deterministic OpenCV preprocessing pipeline
│   ├── wall_detector.py       # Probabilistic Hough line transform & orientation analysis
│   ├── room_extractor.py      # Contour hierarchy, polygon approximation & heuristic scoring
│   ├── floorplan_parser.py    # Headless schema assembler and updater
│   └── room_editor.py         # Human correction, manual room creation & bounds validation
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py       # Architectural overlays, color-coded walls & room badges
│   ├── validation.py          # Shapely polygon topology, coordinates & scale validation
│   └── export.py             # Safe JSON export, disk persistence & import
│
├── assets/
│   └── logo/
│       └── spaze_logo.svg     # Modern geometric vector branding logo
│
├── sample_data/
│   ├── sample_floorplan.png   # Standard 2-bedroom test fixture (800x600 px)
│   ├── generate_sample.py     # Script used to generate the test fixture
│   └── LICENSE_AND_SOURCE.md  # Provenance and CC0 license documentation
│
└── tests/
    ├── __init__.py
    ├── test_image_loader.py     # File validation, corrupted headers, size limits
    ├── test_image_processor.py  # Stage outputs, binarization methods, immutability
    ├── test_wall_detector.py    # Hough lines, angle tolerance, empty map handling
    ├── test_room_extractor.py   # Enclosed contours, noise rejection, candidate scoring
    ├── test_floorplan_schema.py # Serialization, scale propagation, status lifecycle
    ├── test_room_editor.py      # Corrections, manual room creation, bounds checks
    └── test_integration.py     # Full end-to-end processing & JSON roundtrip
```

---

## 3. Technology Stack & Prerequisites

- **Python Version:** Python 3.11 or 3.12 (64-bit)
- **Streamlit:** Modern reactive web UI framework
- **OpenCV (`opencv-python-headless`):** Deterministic computer vision operations
- **NumPy:** Multidimensional array handling and vector operations
- **Pillow (PIL):** Format integrity verification and memory-safe image decoding
- **Matplotlib:** Color palettes and coordinate mapping
- **Shapely:** Computational geometry and polygon topology validation
- **Pytest:** Comprehensive automated testing suite

---

## 4. Installation & Setup Instructions

### Step 1: Clone or Navigate to the Workspace
```powershell
cd D:\Spaze
```

### Step 2: Create a Dedicated Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(On Command Prompt: `.\.venv\Scripts\activate.bat`)*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 5. Execution & Verification

### Running the Streamlit Application
Launch the web interface locally:
```powershell
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

### Running the Automated Test Suite
Execute all 37 unit and integration tests:
```powershell
pytest -v
```
Or in quiet summary mode:
```powershell
python -m pytest -q
```

---

## 6. How to Use the Application

### 1. Uploading a Floor Plan
- **Drag-and-Drop:** Upload any `.png`, `.jpg`, or `.jpeg` architectural drawing into the upload zone.
- **One-Click Sample:** Click **"📂 Load Sample Test Fixture"** to instantly test the 2-bedroom sample fixture (`sample_floorplan.png`).

### 2. Inspecting Computer Vision Stages
Click across the inspection tabs:
- **🏷️ Annotated Rooms:** Overlays detected candidate boundaries with semi-transparent fills and status badges.
- **🧱 Wall Candidates:** Visualizes detected rectilinear wall segments (Cyan = Horizontal, Magenta = Vertical, Gold = Diagonal).
- **🔲 Threshold / Binary:** Inverted binary wall mask where structural boundaries are white (255) and spaces are black (0).
- **⚡ Canny Edges:** High-contrast gradient edge map.
- **🔬 Morphology:** Morphological closing output bridging doorway gaps and small line breaks.
- **🔘 Grayscale & Source:** Clean normalized grayscale and source inspection.

### 3. Calibrating Physical Scale
1. Expand the **"📏 Physical Scale Calibration"** panel.
2. Enter a known pixel distance (e.g. `680.0` px) and its real-world length in meters (e.g. `13.6` m).
3. Click **"Apply Scale"**.
4. The application computes the exact ratio (e.g. `50.00 px/m`) and automatically updates all candidate and confirmed rooms with physical widths, heights, and square meter areas.
5. If no scale is provided, the system strictly marks dimensions as uncalibrated pixels.

### 4. Human-in-the-Loop Room Corrections
- **Select a Room:** Pick any candidate from the dropdown.
- **Edit Details:** Rename the label (e.g., "Master Suite"), assign an architectural room type (Bedroom, Kitchen, Living Room, etc.), and adjust coordinates if needed.
- **Status Lifecycle:**
  - Click **"✅ Confirm Room"** to mark it verified (`is_user_confirmed = True`).
  - Click **"❌ Reject Room"** to discard non-room artifacts or corridors (`detection_status = "Rejected"`).
- **Add Room Manually:** Expand **"➕ Add Room Manually"** to draw a custom room that the automatic vision pipeline missed (e.g. an outdoor balcony or utility closet).

### 5. Exporting Structured Data
- Under **"💾 Structured Floor-Plan Data & Export"**, click **"📥 Download Structured Floor-Plan (JSON)"**.
- The exported JSON contains exact pixel coordinates, physical measurements (if calibrated), wall line vectors, and processing metadata.

---

## 7. Computer Vision Processing Pipeline

```
Uploaded Image (Bytes)
      ↓
Image Validation (Magic bytes signature & PIL verify)
      ↓
Standardized RGB Array Decoding
      ↓
Grayscale Conversion (`COLOR_RGB2GRAY`)
      ↓
Denoising (Bilateral Filter or Gaussian Blur)
      ↓
Adaptive / Otsu Binarization (`ADAPTIVE_THRESH_GAUSSIAN_C`)
      ↓
Canny Edge Detection (`cv2.Canny`)
      ↓
Morphological Closing (`cv2.MORPH_CLOSE` with rectangular kernel)
      ↓
Hough Wall Line Extraction (`cv2.HoughLinesP` + orientation classification)
      ↓
Enclosed Space Contour Detection (`cv2.findContours` with `RETR_CCOMP`)
      ↓
Douglas-Peucker Polygon Simplification & Shapely Topology Verification
      ↓
Heuristic Quality Scoring (Rectangularity, Solidity, Aspect Ratio)
      ↓
Initial Status Assignment (Candidate / Needs Review)
      ↓
Interactive Human Calibration & Correction
      ↓
Structured FloorPlan Object (`models.floorplan_schema.FloorPlan`)
      ↓
Validated JSON Export
```

### Heuristic Scoring Methodology
Quality scores are computed objectively without fabricating probabilities:
$$\text{Quality Score} = 0.45 \cdot R + 0.35 \cdot S + 0.20 \cdot A$$
- **$R$ (Rectangularity):** Ratio of contour area to bounding box area ($\frac{\text{area}}{w \cdot h}$).
- **$S$ (Solidity):** Ratio of contour area to convex hull area ($\frac{\text{area}}{\text{convex hull area}}$).
- **$A$ (Aspect Ratio Score):** Measure of dimensional proportion, rewarding standard architectural rooms in the $[0.5, 2.0]$ ratio range.

Every room carries its full explanation string:
`Heuristic quality: 0.88 (rect=0.92, solid=0.98, aspect=1.05, vertices=4).`

---

## 8. Known Limitations & Edge Cases

1. **Incomplete or Open-Plan Layouts:** If an architectural layout has wide open archways without walls or doors, contour detection may merge connected rooms into a single large candidate. The user can manually split or adjust room bounds in the editor.
2. **Heavy Text & Furniture Hatching:** Hand-drawn blueprints with heavy hatching, text annotations across walls, or complex furniture blocks can introduce false wall segments or fragmented contours. Adjusting Canny thresholds and Hough parameters in the sidebar mitigates this.
3. **Scale Ambiguity:** Real-world physical dimensions cannot be guessed from pixels alone. The user must calibrate scale via a known measurement line or wall length.

---

## 9. Phase 2 Readiness (Contract for Generative AI CVAE)

The structured JSON output produced by Phase 1 matches the requirements for the Phase 2 Conditional Variational Autoencoder:

```json
{
  "image_width": 800,
  "image_height": 600,
  "coordinate_system": "top_left_origin",
  "scale_information": {
    "is_calibrated": true,
    "pixels_per_meter": 50.0,
    "reference_length_meters": 13.6
  },
  "confirmed_rooms": [
    {
      "room_id": "room_01",
      "room_type": "Bedroom",
      "label": "Master Bedroom",
      "x": 420.0,
      "y": 60.0,
      "width": 320.0,
      "height": 240.0,
      "polygon": [[420.0, 60.0], [740.0, 60.0], [740.0, 300.0], [420.0, 300.0]],
      "coordinate_unit": "pixel",
      "detection_status": "User Confirmed",
      "confidence_score": 0.95,
      "is_user_confirmed": true,
      "physical_width": 6.4,
      "physical_height": 4.8,
      "physical_area": 30.72
    }
  ]
}
```

The Phase 2 synthetic layout generator and CVAE can directly ingest `confirmed_rooms` (or active `candidate_rooms`) as normalized spatial vectors $[x, y, w, h, \text{type\_id}]$ without needing to re-parse raw images or depend on Streamlit.
=======
# Spaze
>>>>>>> 2ebb3d169f2d2bb756252b3c46f6f148cf6d0f5a
