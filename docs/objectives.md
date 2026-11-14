# SPAZE — Project Objectives

## 1. Primary Research Goal
To design, develop, rigorously evaluate, and deliver **SPAZE**, an end-to-end local generative AI and computational geometry platform capable of parsing 2D floor-plan raster drawings, generating structurally viable layout alternatives under explicit style conditioning, and synthesizing interactive, watertight 3D architectural models.

---

## 2. Specific Technical Objectives

### Objective 1: Deterministic Local Computer Vision Extraction (Phase 1)
- Develop an edge-heuristic computer vision pipeline using OpenCV that operates completely offline without cloud dependencies.
- Implement progressive multi-stage image processing: grayscale conversion, bilateral denoising, adaptive Gaussian binarization, Canny edge detection, and morphological kernel operations.
- Extract candidate linear wall vectors using the Probabilistic Hough Line Transform ($R^2 \ge 0.85$ correlation).
- Extract closed room candidate contours, approximate their geometric boundaries via polygon simplification, and compute bounding coordinates $(x, y, w, h)$.
- Provide an interactive, human-in-the-loop room editor enabling label correction, type reclassification, dimension editing, and real-world scale calibration (pixels-to-meters).

### Objective 2: Preference-Conditioned Generative AI Engine (Phase 2)
- Formulate architectural room layout generation as a Conditional Variational Autoencoder (CVAE) problem parameterized in PyTorch.
- Structure spatial floor plans as normalized fixed-dimensional room tensors $\mathbf{x} \in \mathbb{R}^{M \times D}$ where $M$ is the maximum room count and $D$ represents normalized spatial parameters ($x, y, w, h$, one-hot type embedding).
- Condition generation on architectural design vectors $\mathbf{c} \in \{\text{Space-Efficient}, \text{Open Layout}, \text{Privacy-Focused}\}$.
- Design an 8-layer deep MLP encoder-decoder architecture with batch normalization, LeakyReLU activations, and reparameterization trick sampling from an 8-dimensional Gaussian latent space $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.
- Train the model using a composite Evidence Lower Bound (ELBO) loss combining Smooth L1 reconstruction loss and Kullback-Leibler (KL) divergence with cyclic annealing.
- Achieve layout synthesis inference latency under 20 ms with a spatial diversity score ($1 - \text{IoU}$) exceeding 0.70.

### Objective 3: Watertight Procedural 3D Geometry Synthesis (Phase 3)
- Bridge 2D layout vectors to 3D manifold meshes via computational geometry libraries (Trimesh, Shapely, Mapbox-Earcut).
- Extrude exterior building envelope walls ($0.20\,\text{m}$ thickness) and interior partition walls ($0.12\,\text{m}$ thickness) with customizable ceiling heights ($2.80\,\text{m}$).
- Automatically calculate adjacent boundary wall segments and carve physical door openings ($0.9\,\text{m} \times 2.1\,\text{m}$) and exterior window apertures ($1.2\,\text{m} \times 1.4\,\text{m}$) using constructive solid geometry.
- Procedurally generate semantic room floors with realistic PBR-like material finishes (light oak hardwood for living areas, grey porcelain tile for bathrooms, polished concrete for corridors).
- Procedurally place dimensionally accurate, non-colliding 3D furniture proxies (beds, dining sets, kitchen counters, sofas).
- Export complete scenes to industry-standard binary glTF 2.0 (`.glb`) with full texture/vertex color preservation.

### Objective 4: Scientific Research Evaluation & UI/UX Transformation (Phase 4)
- Implement quantitative computer vision metrics: Precision, Recall, and F1 score for wall line detection, and Mean Room Intersection-over-Union (IoU) against verified ground-truth CAD annotations.
- Establish an empirical comparative benchmark against a deterministic procedural packing baseline under identical spatial constraints.
- Provide full training telemetry, empirical 45-epoch convergence curves, and preference sensitivity diagnostic matrices.
- Transform the user interface into a refined architectural SaaS workspace using a clean light design system (`#F7F7F2` canvas, `#FFFFFF` cards, `#4263EB` cobalt accents).
- Implement a comprehensive Project Export Hub delivering GLB models, JSON schemas, 3D manifests, PNG blueprints, and CSV telemetry in one click.
