# SPAZE — Future Scope & Research Horizons

## 1. Architectural & Algorithmic Extensions

### 1.1 Multi-Storey Vertical Stacking & Circulation
- Extend the structured `FloorPlan` schema into a hierarchical `BuildingModel` supporting multi-level floors ($L_0, L_1, \dots, L_k$).
- Implement procedural stairwell cutouts and vertical shafts, guaranteeing that structural load-bearing walls and plumbing stacks align vertically across consecutive storeys.

### 1.2 Graph Neural Networks & Transformer-Based Layout Generation
- Explore Graph Neural Network (GNN) and Diffusion Transformer (DiT) architectures for relational room graph modeling.
- Represent architectural programs as heterogeneous scene graphs where nodes represent rooms (with semantic attributes) and edges encode functional topological adjacencies (e.g., "living room adjacent to balcony", "kitchen adjacent to dining").

### 1.3 Energy & Daylighting Simulation Integration
- Integrate computational fluid dynamics (CFD) and solar ray-tracing analysis directly into Stage 04.
- Calculate passive thermal comfort, natural cross-ventilation corridors, and daylight autonomy percentages ($DA_{300}$) based on procedural window aperture locations and geographic solar orientation.

---

## 2. Platform & Toolchain Integrations

### 2.1 Native BIM & Industry-Standard IFC Export
- Extend `geometry3d/exporter.py` with Industry Foundation Classes (IFC 4.0 / IFC 4.3) serialization using `IfcOpenShell`.
- Convert procedural walls, slabs, doors, and windows into native `IfcWallStandardCase`, `IfcSlab`, `IfcDoor`, and `IfcWindow` semantic entities, enabling direct imports into Autodesk Revit, Graphisoft ArchiCAD, and Bentley MicroStation.

### 2.2 Augmented Reality (AR) & Virtual Reality (VR) Inhabitation
- Implement real-time WebXR support directly within the browser, allowing clients and architects to enter and walk through generated 3D floor plans at 1:1 human scale using Meta Quest or Apple Vision Pro headsets.

### 2.3 Structural Engineering Load-Bearing Analysis
- Integrate lightweight finite element analysis (FEA) modules to evaluate structural integrity, identifying cantilevered spans exceeding safety limits and automatically placing structural columns and lintel supports.
