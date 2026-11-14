# SPAZE — Live Demonstration Script (5 to 7 Minutes)

## Presentation Objective
This script provides an engaging, professional, time-calibrated walkthrough designed for university evaluators, academic reviewers, and architectural stakeholders.

---

### Phase 1: Problem Introduction & Project Overview (1:00 min)
- **Speaker Action**: Open browser to `http://localhost:8501`. Display **Stage 00: Workspace Overview**.
- **Spoken Narration**:
  > *"Good morning, esteemed committee members. Today, we are presenting **SPAZE** — an AI-powered architectural design platform that transforms 2D raster floor-plan sketches into preference-conditioned architectural layouts and interactive, watertight 3D digital environments.*
  >
  > *In conventional architectural practice, converting a 2D sketch into a 3D model takes hours of tedious retracing in CAD packages. Meanwhile, recent black-box image diffusion models hallucinate non-physical, warped walls with zero geometric accuracy.*
  >
  > *SPAZE bridges this gap through a hybrid approach: local computer vision parses blueprints, a Conditional Variational Autoencoder synthesizes diverse layouts in under 10 milliseconds, and computational geometry guarantees 100% physical feasibility and watertight 3D mesh deliverables. Let's see it in action."*

---

### Phase 2: Computer Vision & Scale Calibration (1:30 min)
- **Speaker Action**: Click **"01 — Floor Plan Studio"** in the sidebar. Click **"📂 Load Sample Test Fixture"**.
- **Spoken Narration**:
  > *"Here in Stage 01, we ingest an architectural drawing. Completely locally—with zero cloud dependencies—SPAZE executes a multi-stage OpenCV pipeline: bilateral filtering, adaptive Gaussian thresholding, Canny edge detection, and Hough transforms.*
  >
  > *Switching through these inspection tabs, you can see how our algorithms separate noisy paper grain from structural wall lines. Furthermore, our contour extraction identifies closed rooms, assigning initial bounding boxes.*
  >
  > *Notice this physical scale calibration tool: by defining a known distance of 13.6 meters, SPAZE calibrates the entire plan from raw pixels to metric units, allowing architects to review real-world square meter areas and make manual corrections with human-in-the-loop verification."*

---

### Phase 3: Generative AI Layout Synthesis (1:30 min)
- **Speaker Action**: Click **"👉 Continue to AI Generation (Stage 02)"**. Select **"🛋️ Open Layout"**. Click **"🚀 Generate 3 Alternative Layouts"**.
- **Spoken Narration**:
  > *"In Stage 02, our trained PyTorch Conditional Variational Autoencoder takes over. We condition generation on specific architectural design preferences: Space-Efficient, Open Layout, or Privacy-Focused.*
  >
  > *In just 8.4 milliseconds on this standard laptop CPU, the CVAE samples an 8-dimensional Gaussian latent space, producing three distinct, geometrically validated layout alternatives.*
  >
  > *Notice the side-by-side comparison: our model achieves a high layout diversity score of 0.72 while maintaining 100% geometric validity. Let's select Alternative 2—an open-concept design—and proceed to 3D generation."*

---

### Phase 4: Procedural 3D Generation & WebGL Inspection (1:30 min)
- **Speaker Action**: Click **"👉 Proceed to 3D Visualization (Stage 03)"**. Orbit, zoom, and toggle layers in the Plotly 3D canvas.
- **Spoken Narration**:
  > *"Here in Stage 03, SPAZE translates the 2D layout into a complete, watertight 3D building model. Notice this is not a pre-rendered video or static image—this is a live, hardware-accelerated WebGL mesh.*
  >
  > *Our computational geometry engine procedurally extrudes the exterior envelope and interior partition walls, calculates adjacent room boundaries, and carves genuine physical door and window apertures with constructive solid geometry.*
  >
  > *Floor slabs are tessellated with realistic material finishes, and non-colliding procedural furniture proxies are placed. We can orbit, zoom, adjust ceiling heights from 2.8 to 3.5 meters in real time, and immediately export the model as a production-ready binary GLB file ready for Blender, Unreal Engine, or Unity."*

---

### Phase 5: Scientific Evaluation & Conclusion (1:00 min)
- **Speaker Action**: Click **"04 — Evaluation Lab"** and **"05 — Project Export Hub"**.
- **Spoken Narration**:
  > *"Finally, in our Evaluation Lab, we provide full academic transparency. Against verified CAD ground truth, our computer vision pipeline achieves 88.2% wall precision and 0.814 mean room IoU.*
  >
  > *Crucially, our head-to-head empirical benchmark against a deterministic procedural baseline proves that while both systems achieve 100% geometric validity, our CVAE delivers +297% higher spatial diversity with superior topological adjacency preservation.*
  >
  > *With 88 automated passing unit tests and one-click multi-format exports, SPAZE demonstrates how generative AI and computational geometry unite into a robust, local-first architectural CAD tool. Thank you, and we welcome your questions."*
