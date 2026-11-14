# SPAZE — System Limitations & Constraints

## 1. Boundary & Geometry Scope Limitations
1. **Single-Story Focus**: The current procedural 3D generation engine is specialized for single-level floor plans and residential apartments. Multi-story vertical stacking, stairwell cutouts, and elevator shafts are not procedurally resolved in this release.
2. **Rectilinear Bias in Generative Engine**: The CVAE tensor representation encodes bounding boxes $(x, y, w, h)$. While the computer vision and 3D mesh engines support arbitrary polygons and diagonal walls, the generative layout engine approximates non-orthogonal layouts to rectilinear room envelopes.
3. **Curvilinear Geometry**: Organic, non-linear architectural walls (e.g., circular rotundas, splines, elliptical facades) are approximated via polygonal linear segments rather than true NURBS surfaces.

---

## 2. Computer Vision Preprocessing Constraints
1. **Raster Resolution Sensitivity**: Extremely low-resolution blueprints (under $400 \times 300$ pixels) or heavily compressed JPEG images exhibiting severe blocking artifacts can degrade wall line detection precision below $70\%$.
2. **Dense Annotation Noise**: Blueprints cluttered with dense dimension chains, mechanical plumbing symbols, electrical tags, and interior text labels overlapping wall edges require human-in-the-loop manual verification in Stage 01.
3. **Furniture Symbol Interference**: Hand-drawn floor plans containing solid black furniture silhouettes (e.g., solid bathtubs or beds) can occasionally be detected as structural wall candidates if their perimeter gradient mimics wall lines.

---

## 3. Generative Model Limitations
1. **Fixed Maximum Room Capacity**: The current CVAE input tensor is dimensioned for a maximum room capacity of $M = 10$ rooms. Very large commercial buildings (e.g., hospitals, shopping centers with 50+ rooms) exceed the tensor capacity and require spatial hierarchical decomposition.
2. **Dataset Domain Specificity**: The model weights (`spaze_cvae_best.pth`) were trained on residential typologies (1BHK, 2BHK, 3BHK, Studios). Generating specialized commercial facilities (e.g., open-plan offices, cleanrooms, data centers) requires fine-tuning on domain-specific training splits.

---

## 4. 3D Model & Rendering Constraints
1. **Procedural Furniture Proxies**: Furniture elements are synthesized as clean geometric bounding proxies (e.g., box beds with nightstands, dining tables with cylinder chairs) rather than high-polygon photorealistic CAD assets to maintain sub-20 ms mesh synthesis and instant WebGL rendering.
2. **Lighting & Raytracing**: The interactive viewer utilizes WebGL diffuse/specular real-time shading via Plotly Mesh3d. Photorealistic global illumination, soft shadows, and raytraced reflections require exporting the binary `.glb` model into specialized renderers (e.g., Blender Cycles or Unreal Engine Lumen).
