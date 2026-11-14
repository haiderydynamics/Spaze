# SPAZE — Synthetic Architectural Dataset Methodology

## 1. Rationale for Synthetic Architectural Procedural Datasets
Real-world CAD datasets (such as R-PLAN or LIFULL) present significant research barriers:
1. **Restrictive Commercial Licensing**: Many CAD repositories prohibit open-source academic reproduction.
2. **Annotation Inconsistencies**: Non-standardized wall definitions, overlapping label artifacts, and absence of physical scale calibrations.
3. **Imbalanced Preference Classes**: Real-world archives are skewed toward conventional layouts with virtually no explicit annotations for "Open Layout" or "Privacy-Focused" programmatic intent.

To ensure 100% reproducible, verifiable research without legal or empirical barriers, SPAZE incorporates an automated procedural architectural dataset generator (`cvae/dataset.py`).

---

## 2. Dataset Synthesis Algorithm
The generator constructs realistic residential floor plans governed by physical architectural typologies:
- **1-BHK (Single Bedroom, Hall, Kitchen)**: Compact footprints, shared living-dining spaces.
- **2-BHK (Two Bedrooms, Hall, Kitchen)**: Standard residential unit with master suite and guest bedroom.
- **3-BHK (Three Bedrooms, Hall, Kitchen, Multiple Baths)**: Suburban apartment program with compartmentalized private zones.
- **Studio Units**: Open-plan urban micro-apartments.

```mermaid
flowchart TD
    A[Dataset Initialization] --> B[Sample Building Footprint W x H]
    B --> C[Select Architectural Typology: 1BHK, 2BHK, 3BHK, Studio]
    C --> D[Sample Preference Condition: Space-Efficient, Open, Privacy]
    D --> E[Subdivide Canvas via Recursive Spatial Partitioning]
    E --> F[Inject Boundary & Wall Clearances]
    F --> G[Assign Semantic Room Types & Labels]
    G --> H[Check Geometric Adjacency & Non-Overlap]
    H --> I[Normalize Coordinates into Tensor X in R^{M x 8}]
    I --> J[Save to Partitioned Train / Val / Test Splits]
```

---

## 3. Dataset Distribution & Splitting Strategy
The dataset comprises $N = 1{,}200$ procedurally synthesized architectural floor plans generated under strict seed controls (`seed=42`):

| Data Split | Ratio | Sample Count | Conditioning Distribution | Average Room Count |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 80% | 960 layouts | 320 Space-Efficient / 320 Open / 320 Privacy | 5.4 rooms |
| **Validation Set** | 10% | 120 layouts | 40 Space-Efficient / 40 Open / 40 Privacy | 5.3 rooms |
| **Test Set (Held-Out)** | 10% | 120 layouts | 40 Space-Efficient / 40 Open / 40 Privacy | 5.5 rooms |

---

## 4. Architectural Domain Constraints Enforced During Generation
1. **Minimum Room Dimensions**:
   - Living Rooms: $\ge 3.0\,\text{m} \times 3.0\,\text{m}$ ($9.0\,\text{m}^2$)
   - Bedrooms: $\ge 2.8\,\text{m} \times 2.8\,\text{m}$ ($7.84\,\text{m}^2$)
   - Kitchens: $\ge 2.0\,\text{m} \times 2.2\,\text{m}$ ($4.4\,\text{m}^2$)
   - Bathrooms: $\ge 1.5\,\text{m} \times 1.8\,\text{m}$ ($2.7\,\text{m}^2$)
2. **Boundary Containment**: All room bounding boxes must lie strictly inside the external perimeter wall envelope with a minimum boundary margin of $0.20\,\text{m}$.
3. **Aspect Ratio Enforcement**: Room bounding boxes maintain aspect ratios between $1.0$ (square) and $2.2$ to avoid unbuildable needle-like geometry.
4. **Circulation Verification**: At least one primary room (Living Room or Central Hall) must provide immediate access to secondary zones without requiring traversal through private bathrooms.
