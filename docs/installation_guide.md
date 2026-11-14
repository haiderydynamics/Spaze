# SPAZE — Installation & Environment Setup Guide

## 1. System Requirements

### Hardware Requirements
- **Operating System**: Windows 10/11 (64-bit), macOS 12+ (Apple Silicon or Intel), or Linux (Ubuntu 20.04+).
- **Processor (CPU)**: Quad-core x86_64 or ARM64 processor (e.g., Intel Core i5/i7, AMD Ryzen 5/7, Apple M1/M2/M3).
- **Memory (RAM)**: Minimum 8 GB RAM (16 GB recommended for large 3D meshes).
- **Storage**: Minimum 2 GB of free disk space.
- **GPU (Optional)**: CUDA-compatible GPU supported but **not required**. All modules run in real time on CPU.

### Software Prerequisites
- **Python**: Version 3.10, 3.11, 3.12, or 3.13 (64-bit).
- **Package Manager**: `pip` (Python package installer).
- **Git** (for cloning the repository).

---

## 2. Step-by-Step Installation

### Step 1: Clone or Navigate to the Workspace
Open a PowerShell, Bash, or Command Prompt terminal:
```bash
cd /path/to/Staze-master
```

### Step 2: Create and Activate a Clean Python Virtual Environment (Recommended)
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade Pip
```bash
python -m pip install --upgrade pip
```

### Step 4: Install Locked Project Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Environment & Dependency Smoke-Testing
Execute the automated smoke-test suite to verify all computer vision, deep learning, and 3D graphics libraries are properly installed and linked:
```bash
python smoke_test.py
```

Expected output:
```
[INFO] PyTorch version: 2.x.x
[INFO] OpenCV version: 4.x.x
[INFO] Trimesh version: 4.x.x
[INFO] Plotly version: 5.x.x
[INFO] Shapely version: 2.x.x
[SUCCESS] All SPAZE environment dependencies successfully validated!
```

---

## 4. Run Automated Unit & Regression Tests
To verify all algorithmic modules (88 unit tests across CV, CVAE, 3D Geometry, and Evaluation):
```bash
pytest
```
Expected output:
```
======================== 88 passed in ~55s ========================
```

---

## 5. Launch the SPAZE Application
Start the unified Streamlit application on your local machine:
```bash
streamlit run app.py
```
The application will automatically launch in your default web browser at:
`http://localhost:8501`
