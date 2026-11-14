# SPAZE — Problem Statement

## 1. Background and Context
Architectural floor-plan layout design is the foundational phase of any building lifecycle. During early architectural programming, designers receive customer briefs comprising functional requirements (e.g., number of bedrooms, open living concepts, circulation paths) and spatial boundary envelopes (e.g., plot boundary, structural exterior walls). 

Historically, exploring layout variations requires architects to draft multiple 2D schematics manually in Computer-Aided Design (CAD) packages like AutoCAD, followed by painstaking extrusion in 3D modeling packages like SketchUp, Revit, or Rhino. This process is inherently iterative, computationally fragmented, and labor-intensive.

## 2. Core Deficiencies in Current Architectural Workflows

### 2.1 The 2D-to-3D Semantic Gap
Converting a 2D floor-plan image or blueprint sketch into a 3D building model is plagued by the loss of semantic data:
- **Raster Ambiguity**: Pixel blueprints do not inherently encode semantics—walls are simply raster pixel clusters, text labels clash with dimension markers, and door arcs blend into structural boundaries.
- **Manual Retracing**: Architectural engineers spend 40% to 60% of schematic drafting time simply retracing 2D raster images into 3D vectors.
- **Aperture Coordination**: Creating physical door and window openings in 3D walls requires manual boolean subtractions or precise coordinate alignment, which are prone to topological leaks and non-manifold mesh artifacts.

### 2.2 Inflexibility of Traditional Procedural Generative Systems
Prior attempts at automated architectural layout generation rely heavily on heuristic algorithms, such as:
- Linear Programming and Mixed-Integer Quadratic Optimization.
- Genetic Algorithms and Evolutionary Layout Optimization.
- Rule-based cellular automata and packing heuristics.

While these approaches guarantee rigid adherence to geometric constraints, they exhibit severe drawbacks:
1. **Combinatorial Explosion**: As room counts and adjacency graphs expand, procedural solvers scale exponentially ($O(n!)$ or $O(2^n)$), leading to prohibitive computational latency.
2. **Deterministic Monotony**: Rule-based packing consistently produces identical, rigid rectilinear layouts that lack organic architectural sensibility and cannot navigate complex human preferences.
3. **Rigid Inflexibility**: Modifying design goals requires rewriting handcrafted rule weights rather than sampling continuous latent spaces.

### 2.3 The Hallucination Problem of Black-Box Deep Learning
Recent advances in generative diffusion models (e.g., Stable Diffusion, Midjourney) and Large Vision Models (LVMs) produce visually striking architectural renderings. However, they suffer from fatal engineering flaws when applied to architectural engineering:
- **Zero Geometric Feasibility**: Neural pixel generators generate non-rectilinear, warped walls that cannot be manufactured or built in physical space.
- **Lack of Structured Semantics**: Pixel synthesis creates an image of a floor plan without vector polygons, coordinate bounding boxes, or adjacency graphs.
- **Inability to Export to CAD/BIM**: Architectural engineering workflows require industry-standard glTF/GLB, IFC, or JSON vector representations, which pixel diffusion models cannot deliver.

## 3. The Research Problem
There exists a critical need for an integrated system that can:
1. Parse arbitrary 2D architectural blueprint images locally without cloud latency or security risks.
2. Formulate architectural layout synthesis as a continuous probabilistic learning task conditioned on explicit style objectives.
3. Guarantee 100% geometric validity and boundary containment through computational geometry verification.
4. Procedurally construct watertight, physical 3D meshes with real-world metric dimensions and standard CAD/GLB deliverables.
