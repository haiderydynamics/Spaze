# SPAZE — Implementation Details & Technical Specifications

## 1. System Technology Stack
SPAZE is engineered natively in Python 3.10+ without external node.js or browser daemon requirements.

| Layer / Subsystem | Technology | Version | Architectural Purpose |
| :--- | :--- | :--- | :--- |
| **Framework Runtime** | Python (CPython) | 3.10+ (Tested 3.13) | Core language runtime |
| **UI Application Host** | Streamlit | 1.30+ | Local reactive single-page application |
| **Computer Vision** | OpenCV (opencv-python) | 4.8+ | Image preprocessing, Hough lines, contours |
| **Deep Learning** | PyTorch (torch) | 2.1+ | CVAE tensor neural network architecture |
| **Linear Algebra** | NumPy | 1.24+ | Vector calculations, matrix transformations |
| **Computational Geometry** | Shapely | 2.0+ | 2D polygon buffering, intersections, clipping |
| **3D Mesh Data Structures** | Trimesh | 4.0+ | Watertight 3D scene construction & manifold validation |
| **Polygon Triangulation** | Mapbox-Earcut | 1.1+ | Robust 2D floor polygon triangulation |
| **3D WebGL Visualization** | Plotly (plotly.graph_objects) | 5.18+ | Native hardware-accelerated Mesh3d canvas |
| **CAD Model Export** | pygltflib / trimesh.exchange | Latest | Binary glTF 2.0 (.glb) serialization |
| **Testing & CI/CD** | Pytest | 7.4+ | Automated unit, regression, and integration tests |

---

## 2. Neural Network Architecture Specification
The generative core is instantiated by the `SpazeCVAE` class (`cvae/model.py`), an 8-layer deep symmetric Multi-Layer Perceptron (MLP).

### 2.1 Encoder Architecture ($q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{c})$)
- **Input Dimension**: $80 + 3 = 83$ (Flattened spatial room tensor $10 \times 8 = 80$, concatenated with 3-dimensional condition vector $\mathbf{c}$).
- **Layer 1**: Linear($83 \to 256$), BatchNorm1d(256), LeakyReLU($\alpha=0.2$), Dropout($p=0.15$)
- **Layer 2**: Linear($256 \to 128$), BatchNorm1d(128), LeakyReLU($\alpha=0.2$), Dropout($p=0.15$)
- **Layer 3**: Linear($128 \to 64$), BatchNorm1d(64), LeakyReLU($\alpha=0.2$)
- **Latent Projection**:
  - Mean ($\boldsymbol{\mu}$): Linear($64 \to 8$)
  - Log-Variance ($\log \boldsymbol{\sigma}^2$): Linear($64 \to 8$)

### 2.2 Decoder Architecture ($p_\theta(\mathbf{x} \mid \mathbf{z}, \mathbf{c})$)
- **Latent Input Dimension**: $8 + 3 = 11$ (Sampled latent vector $\mathbf{z} \in \mathbb{R}^8$ concatenated with conditioning vector $\mathbf{c} \in \mathbb{R}^3$).
- **Layer 1**: Linear($11 \to 64$), BatchNorm1d(64), LeakyReLU($\alpha=0.2$)
- **Layer 2**: Linear($64 \to 128$), BatchNorm1d(128), LeakyReLU($\alpha=0.2$), Dropout($p=0.15$)
- **Layer 3**: Linear($128 \to 256$), BatchNorm1d(256), LeakyReLU($\alpha=0.2$), Dropout($p=0.15$)
- **Output Projection**: Linear($256 \to 80$), Sigmoid Activation (constraining all spatial coordinates to $[0, 1]$).
- **Total Trainable Parameters**: **113,872 parameters** (Model checkpoint footprint: **88.6 KB** on disk).

---

## 3. Training Protocol & Hyperparameters
- **Optimization Algorithm**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, Weight Decay $= 10^{-5}$)
- **Initial Learning Rate**: $\eta = 10^{-3}$ with ReduceLROnPlateau scheduler (decay factor $= 0.5$, patience $= 5$ epochs).
- **Batch Size**: 32
- **Training Epochs**: 45 epochs with early stopping monitoring validation ELBO loss.
- **KL Divergence Annealing**: Linear cyclical annealing from $\beta = 0.0$ to $\beta = 0.01$ over the first 10 epochs to prevent posterior collapse (where latent units become inactive noise).

---

## 4. 3D Procedural Synthesis Algorithmic Rules
1. **Wall Classification**:
   - Exterior Envelope: Walls bordering the external building polygon. Extruded with thickness $T_{\text{ext}} = 0.20\,\text{m}$.
   - Interior Partition: Boundary segments separating two adjacent rooms. Extruded with thickness $T_{\text{int}} = 0.12\,\text{m}$.
2. **Door Slicing Algorithm**:
   - For every interior wall segment separating adjacent rooms with length $L \ge 1.4\,\text{m}$, a door opening of width $W_{\text{door}} = 0.90\,\text{m}$ and height $H_{\text{door}} = 2.10\,\text{m}$ is centered along the segment.
   - The wall is decomposed into three sub-volumes: left jamb box, right jamb box, and top lintel box ($Z \in [2.10\,\text{m}, 2.80\,\text{m}]$), guaranteeing manifold topology without non-planar boolean artifacts.
3. **Window Aperture Algorithm**:
   - Exterior walls with length $L \ge 2.0\,\text{m}$ receive window cutouts of width $W_{\text{win}} = 1.20\,\text{m}$, height $H_{\text{win}} = 1.40\,\text{m}$, and sill height $Z_{\text{sill}} = 0.90\,\text{m}$.
   - Decomposed into bottom sill box, left jamb, right jamb, top header, and an inset semi-transparent glass panel ($A = 0.45$).
