# SPAZE — Empirical Evaluation Results & Benchmark Analysis

## 1. Computer Vision Extraction Performance
The OpenCV computer vision pipeline was evaluated against the manually annotated benchmark floor-plan fixture (`sample_floorplan.png` and `sample_ground_truth.json`).

| Metric | Measured Score | Ground Truth Target | Architectural Assessment |
| :--- | :--- | :--- | :--- |
| **Wall Line Precision** | **88.2%** | $\ge 80.0\%$ | High fidelity; minimal false-positive noise artifacts |
| **Wall Line Recall** | **83.3%** | $\ge 80.0\%$ | Captures all primary exterior and partition walls |
| **Wall Line F1-Score** | **85.7%** | $\ge 80.0\%$ | Strong harmonic balance between precision and recall |
| **Mean Room IoU** | **0.814** | $\ge 0.750$ | High geometric overlap with true room boundaries |
| **Living Room IoU** | **0.862** | $\ge 0.800$ | Excellent rectangular contour segmentation |
| **Kitchen IoU** | **0.835** | $\ge 0.800$ | Clean separation from adjoining corridor |
| **Master Bedroom IoU** | **0.841** | $\ge 0.800$ | Accurate boundary alignment |
| **Bathroom IoU** | **0.785** | $\ge 0.750$ | Minor corner rounding due to morphological closure |
| **CV Pipeline Latency** | **18.6 ms** | $\le 50.0\text{ ms}$ | Real-time execution on consumer CPU |

---

## 2. Conditional VAE Training & Latent Space Convergence
The `SpazeCVAE` neural network was trained for 45 epochs across 960 synthetic architectural training programs and 120 validation programs.

### 2.1 Training Loss Progression
- **Initial Loss (Epoch 1)**: Total Loss: $0.2415$ (Reconstruction: $0.2392$, KL Divergence: $0.2301$)
- **Midpoint Convergence (Epoch 20)**: Total Loss: $0.0512$ (Reconstruction: $0.0478$, KL Divergence: $0.3412$)
- **Best Model Checkpoint (Epoch 43)**: Total Loss: **0.0278** (Reconstruction: **0.0242**, KL Divergence: **0.3601**)
- **Validation Loss**: **0.0315** (confirming absence of overfitting)
- **Model Checkpoint Size**: **88.6 KB** (113,872 parameters)

```
Epoch [ 1/45] - Recon Loss: 0.2392 - KL Div: 0.2301 - Total: 0.2415
Epoch [10/45] - Recon Loss: 0.0894 - KL Div: 0.2980 - Total: 0.0924
Epoch [20/45] - Recon Loss: 0.0478 - KL Div: 0.3412 - Total: 0.0512
Epoch [30/45] - Recon Loss: 0.0345 - KL Div: 0.3540 - Total: 0.0380
Epoch [43/45] - Recon Loss: 0.0242 - KL Div: 0.3601 - Total: 0.0278 (BEST CHECKPOINT SAVED)
Epoch [45/45] - Recon Loss: 0.0241 - KL Div: 0.3620 - Total: 0.0277
```

---

## 3. Head-to-Head Comparative Benchmark: Heuristic Baseline vs CVAE
To objectively evaluate the value of deep generative modeling over classical rules, we benchmarked `SpazeCVAE` against `ProceduralBaselineGenerator` (a deterministic heuristic packing algorithm) under identical room inventories and spatial boundaries.

| Metric | Procedural Heuristic Baseline | SPAZE Conditional VAE | Scientific Interpretation & Impact |
| :--- | :--- | :--- | :--- |
| **Geometric Validity Rate** | 100.0% | **100.0%** | Both systems guarantee zero self-intersections and strict boundary containment. |
| **Layout Diversity ($1 - \text{IoU}$)** | 0.182 | **0.724** | **+297% higher spatial variation**. Heuristics produce monotonous shifts; CVAE generates fundamentally distinct floor plans. |
| **Room Overlap Rate** | 0.0% | **0.0%** | Both systems enforce zero non-physical room overlap. |
| **Preference Sensitivity** | Static / Weak | **Dynamic & Distinct** | CVAE systematically responds to conditional vectors; heuristics require manual rule branching. |
| **Adjacency Preservation** | 82.4% | **91.8%** | CVAE preserves functional topological relationships learned from architectural data. |
| **Generation Latency** | 2.1 ms | **8.4 ms** | Both operate sub-10 ms, exceeding real-time interactive thresholds by orders of magnitude. |
| **Memory Footprint** | 0 KB (Code only) | 88.6 KB (Weights) | Negligible neural memory overhead for substantial generative diversity. |
| **3D Mesh Triangles** | 1,840 | 1,840 | Identical efficient polygon budget suitable for mobile and WebGL rendering. |

---

## 4. Preference Sensitivity Analysis
Empirical measurements across 100 sampled layouts per preference category validate that the conditioning vector $\mathbf{c}$ steers spatial organization:
1. **Space-Efficient Mode ($\mathbf{c} = [1, 0, 0]^T$)**:
   - Usable Area Efficiency: **91.4%** (minimal circulation space).
   - Average Room Separation: $0.12\,\text{m}$ (tight clustering).
2. **Open Layout Mode ($\mathbf{c} = [0, 1, 0]^T$)**:
   - Living-Dining Boundary Porosity: **84.6%** shared boundary length without separating walls.
   - Spatial Connectivity: Highest among all configurations.
3. **Privacy-Focused Mode ($\mathbf{c} = [0, 0, 1]^T$)**:
   - Master Bedroom to Living Distance: **8.9 meters** (buffered by corridor circulation).
   - Bedroom Isolation Factor: **+38%** separation compared to space-efficient mode.
