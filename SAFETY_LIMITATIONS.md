# Safety, Ethics, Limitations & Clinical Governance

## 1. Primary Clinical Safety Constraint
**BREASTCARE_CLASSIFICATION_ENABLED=false (Enforced Default)**

In accordance with medical software ethics, diagnostic classification is disabled by default on the server. Web browsers and API clients cannot override or bypass this server-side configuration flag.

---

## 2. Multi-Stage Fake & Unsupported Image Rejection

Modern medical deep learning models often output high-confidence predictions on invalid inputs (e.g. classifying a pet or coffee mug as malignant). BreastCare AI prevents this through 4 sequential defense gates:

1. **Gate 1: File & Format Validation**
   - Rejects corrupted image byte buffers, non-PNG/JPEG payloads, and files exceeding 10 MB.
2. **Gate 2: Modality Discrimination**
   - Employs physics-based heuristics to identify and reject:
     - Consumer photography (faces, selfies, pets, nature).
     - Text screenshots and scanned paper documents.
     - **Chest X-rays** (detected via bilateral symmetrical lung margins and central spinal columns; never classified as mammograms).
3. **Gate 3: Quality Control & Artefact Detection**
   - Filters out extreme motion blur (Laplacian variance < 15.0), severe contrast collapse, near-uniform blank scans, and extreme sensor saturation.
4. **Gate 4: Out-of-Distribution (OOD) Screening**
   - Rejects images deviating from standard acoustic/radiographic spatial distributions.

### Standardized Response Messaging
- **Unsupported / Non-Medical Image:**
  `"Unsupported image. Please upload a valid breast ultrasound or mammogram."`
- **Borderline / Low Quality / OOD Scan:**
  `"Unable to provide a reliable assessment. Specialist review is recommended."`

---

## 3. Interpretability & Explainability Disclaimers
- Feature attention heatmaps produced via **Grad-CAM (Layer4)** represent internal convolutional activations of the deep learning model.
- **Explicit Notice:** Heatmaps are visual mathematical cues. They **do not** delineate verified tumor margins, micro-calcification clusters, or surgical resectability borders.

---

## 4. Regulatory & Legal Notice
- **Not a Medical Device:** BreastCare AI has not been evaluated, cleared, or approved by the US FDA, EMA, MHRA, or other regulatory bodies.
- **Investigational Prototype Only:** This software is intended solely for scientific benchmarking, academic reproduction, and educational demonstrations.

