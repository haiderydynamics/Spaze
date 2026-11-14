# SPAZE — Research Evaluation Methodology

## 1. Experimental Setup & Benchmarking Principles
To satisfy academic rigor for university-level Generative AI and Computer Vision evaluation, SPAZE incorporates objective quantitative benchmarking methodologies.

Crucially, **no metric is fabricated or hallucinated**:
- Computer Vision extraction metrics are calculated against explicitly annotated ground-truth CAD fixtures (`sample_ground_truth.json`).
- If an unannotated image is uploaded, the system explicitly reports `"Not Evaluated — Ground Truth Unavailable"`.
- CVAE generative metrics are computed across a held-out test split of 120 synthesized architectural programs.
- Comparative benchmarks evaluate the CVAE against a deterministic heuristic baseline (`ProceduralBaselineGenerator`) operating under identical room counts and boundary polygons.

---

## 2. Evaluation Metrics Formulation

### 2.1 Computer Vision Metrics
1. **Wall Line Precision ($P_{\text{wall}}$)**:
   $$P_{\text{wall}} = \frac{TP_{\text{walls}}}{TP_{\text{walls}} + FP_{\text{walls}}}$$
   A detected wall segment $w_d$ is classified as a True Positive ($TP$) if there exists a ground-truth wall $w_{gt}$ such that:
   $$\text{HausdorffDistance}(w_d, w_{gt}) \le \tau_{\text{dist}} \quad (18\,\text{px}) \quad \text{and} \quad |\theta_d - \theta_{gt}| \le \tau_{\text{angle}} \quad (10^\circ)$$

2. **Wall Line Recall ($R_{\text{wall}}$)**:
   $$R_{\text{wall}} = \frac{TP_{\text{walls}}}{TP_{\text{walls}} + FN_{\text{walls}}}$$

3. **Wall F1-Score ($F1_{\text{wall}}$)**:
   $$F1_{\text{wall}} = \frac{2 \cdot P_{\text{wall}} \cdot R_{\text{wall}}}{P_{\text{wall}} + R_{\text{wall}}}$$

4. **Mean Room Intersection-over-Union ($\text{IoU}_{\text{room}}$)**:
   For ground-truth room polygon $P_{gt}$ and predicted room polygon $P_{\text{pred}}$:
   $$\text{IoU}(P_{gt}, P_{\text{pred}}) = \frac{\text{Area}(P_{gt} \cap P_{\text{pred}})}{\text{Area}(P_{gt} \cup P_{\text{pred}})}$$
   $$\text{Mean Room IoU} = \frac{1}{|R_{gt}|} \sum_{i=1}^{|R_{gt}|} \max_{j} \text{IoU}(P_{gt, i}, P_{\text{pred}, j})$$

---

### 2.2 Generative AI Metrics
1. **Reconstruction Loss**: Evaluates structural fidelity between the original layout and the autoencoded reconstruction.
2. **Kullback-Leibler Divergence ($D_{\text{KL}}$)**: Quantifies the information divergence between the variational posterior $q_\phi(\mathbf{z} \mid \mathbf{x}, \mathbf{c})$ and standard normal prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$.
3. **Layout Diversity Score ($1 - \text{IoU}$)**: Measures spatial distinction between pairs of generated alternatives ($A_1, A_2$):
   $$\text{Diversity}(A_1, A_2) = 1.0 - \frac{1}{|R|} \sum_{k=1}^{|R|} \text{IoU}(r_k^{(A_1)}, r_k^{(A_2)})$$
4. **Preference Sensitivity**:
   - *Space Efficiency Ratio*: Usable room floor area divided by gross building boundary area:
     $$\eta_{\text{eff}} = \frac{\sum \text{Area}(R_i)}{\text{Area}(\text{Envelope})}$$
   - *Openness Connectivity Score*: Ratio of shared internal boundary wall length without dividing partitions to total perimeter length.
   - *Privacy Isolation Distance*: Average Euclidean topological path distance separating master bedroom entrances from public living areas.
5. **Geometric Validity Rate**: Percentage of generated layouts satisfying 100% of spatial constraints (zero room self-intersections, zero envelope breaches, minimum room dimensions met).
6. **Inference Latency**: Real-time Wall-clock execution time in milliseconds per layout generation on standard consumer hardware.
