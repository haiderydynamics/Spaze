# SPAZE — Project Abstract

## Title
**SPAZE: An End-to-End Generative AI and Computational Geometry Framework for Automated 2D-to-3D Architectural Floor-Plan Synthesis**

---

### Executive Summary
Architectural spatial layout planning and 3D schematic modeling represent time-consuming, iterative bottlenecks in residential design workflows. While human architects synthesize client preferences, local building codes, structural boundaries, and functional adjacencies, translating conceptual 2D raster floor plans into validated 3D building information models traditionally demands hours of manual CAD drafting and 3D modeling. Existing computer vision solutions either rely on brittle edge heuristics without generative capabilities, or employ unconstrained black-box generative models that produce non-watertight, geometrically invalid rooms lacking physical aperture coordination.

This research presents **SPAZE**, a unified, local-first Generative AI and computational geometry platform that seamlessly bridges the continuum from 2D architectural raster sketches to interactive, physically feasible 3D digital environments. 

The SPAZE system operates through four tightly coupled engineering stages:
1. **Foundation & Local Computer Vision Processing**: Employs deterministic OpenCV pipelines (bilateral filtering, adaptive Gaussian thresholding, Canny edge detection, probabilistic Hough transform, morphological closing, and contour-polygon approximation) to extract candidate wall line vectors, detect closed room cells, and establish real-world metric scaling via interactive pixel-to-meter calibration without external API dependencies.
2. **Generative AI Engine (Conditional Variational Autoencoder)**: Implements an 8-layer deep Conditional VAE (`SpazeCVAE`) parameterized with continuous coordinate-dimension representations and categorical one-hot architectural preference embeddings ($\mathbf{c} \in \{\text{Space-Efficient}, \text{Open Layout}, \text{Privacy-Focused}\}$). Trained with an Evidence Lower Bound (ELBO) objective balancing reconstruction Smooth L1 loss and KL divergence, the network samples distinct, high-fidelity spatial variants ($1 - \text{IoU} \ge 0.72$) in under 12 ms on consumer hardware while preserving topological room programs.
3. **Procedural 3D Geometry Synthesis**: Translates verified 2D layouts into watertight manifold polygon meshes using computational geometry algorithms (Trimesh, Mapbox-Earcut triangulation, Shapely 2D spatial analysis). The engine procedurally extrudes interior and exterior wall envelopes, calculates physical boolean wall apertures (doors and windows), installs floor slabs with semantic room finishes, and places bounding-box constrained procedural furniture items.
4. **Interactive Visualization & Research Evaluation**: Renders the generated building in real-time WebGL via native Plotly Mesh3d graphics supporting multi-layer semantic toggling, orbit/pan controls, and camera presets. Furthermore, an integrated empirical Evaluation Lab compares CVAE outputs against ground-truth CAD annotations and a deterministic heuristic baseline, providing multi-format industry exports (glTF/GLB 2.0, JSON schemas, 3D manifests, and PNG blueprints).

SPAZE demonstrates that combining deep generative latent models with rigorous deterministic geometric validation satisfies both design diversity and architectural validity, establishing a foundation for autonomous computer-aided architectural engineering.
