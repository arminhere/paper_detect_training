# Document Corner Detection End-to-End System Plan

This document outlines a practical, end-to-end architecture and implementation plan for building a robust document corner detection system from zero labeled data, meeting the constraints of mobile camera noise, real-world generalization, and near real-time inference.

## User Review Required
> [!IMPORTANT]
> Please review the proposed architecture and training strategy. The core of solving the "zero labeled data" problem relies on **Synthetic Data Generation**. We need to agree on the synthetic pipeline before writing any model code.

## Open Questions
- What is your target inference device? (e.g., specific mobile hardware, web browser, or desktop server processing mobile uploads?) This impacts how aggressively we need to quantize or prune the model.
- Do we need to handle multi-document detection in a single image, or can we assume a single primary document per frame?

## Proposed Architecture and Phases

### Phase 1: The "Zero Data" Solution - Synthetic Data Generation pipeline
Since we have no labeled data, we will synthetically generate an infinite dataset where we mathematically know the exact corner coordinates.

1.  **Backgrounds:** Collect a large dataset of random indoor, outdoor, and textured images (e.g., MS COCO dataset, or plain textures like wood, carpet, tables).
2.  **Foregrounds (Documents):** Generate or collect flat document images. Use Python's `reportlab` or `faker` to generate fake text documents, receipts, and ID cards with varying layouts.
3.  **The Generator:** 
    *   Sample a background and a document.
    *   Apply a random **Perspective Transform** (homography) to the document to simulate steep camera angles.
    *   *The Magic:* Because we applied the transform, we know exactly where the 4 corners mapped to in the final image. These are our perfect labels!
4.  **Data Augmentation (Crucial for Mobile):**
    *   **Lighting/Shadows:** Apply gradient masks to simulate uneven lighting and hard shadows.
    *   **Blur & Noise:** Apply Gaussian blur, motion blur, and Gaussian noise to simulate moving mobile cameras in low light.
    *   **Occlusion:** Paste random shapes or "fingers" over the edges to force the model to learn global context, not just local edge lines.

### Phase 2: Model Selection (Near Real-Time)
We need a model that runs fast but has high spatial awareness. 

> [!TIP]
> **Recommendation:** A lightweight Heatmap-based approach (similar to HRNet or a slim U-Net predicting 4 Gaussian heatmaps). 
> - **Why not direct regression (predicting 8 numbers)?** Regression struggles with steep angles and occlusions because it lacks spatial context.
> - **Why not full segmentation?** Segmentation requires heavy post-processing (contour finding, polygon approximation) which can be brittle if the edge has a shadow. Heatmaps directly output the corner locations.

*   **Backbone:** MobileNetV3-Small or EfficientNet-B0.
*   **Head:** 4-channel output (one heatmap channel for each corner: Top-Left, Top-Right, Bottom-Right, Bottom-Left).

### Phase 3: Training Strategy
1.  **Pre-training:** Train the model on 100,000+ synthetic images. Use Mean Squared Error (MSE) loss against target heatmaps (Gaussian blobs placed at the true corner coordinates).
2.  **Bootstrapping (Active Learning):**
    *   Deploy the synthetically-trained model into a basic Python script.
    *   Take 100-500 pictures with your actual phone in target environments.
    *   Run the model to predict corners.
    *   Manually correct the predictions that are slightly off.
    *   Fine-tune the model on this small set of *real* data. This bridges the "sim-to-real" gap.

### Phase 4: Inference & Post-Processing Pipeline
1.  **Pre-process:** Resize the incoming mobile frame to a fixed square (e.g., 256x256 or 384x384), normalize.
2.  **Inference:** Run the lightweight PyTorch model -> Outputs 4 heatmaps.
3.  **Extract Coords:** Find the `argmax` (x, y coordinate of the highest value) for each of the 4 heatmaps. Apply sub-pixel refinement if extreme accuracy is needed.
4.  **Scale Back:** Multiply the coordinates by the scaling factor to match the original high-res image.
5.  **Perspective Warp:** Use OpenCV `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to flatten and crop the document.

### Phase 5: Deployment
*   Export the PyTorch model to **ONNX**.
*   Run inference using `cv2.dnn.readNetFromONNX()` or `onnxruntime`. This ensures high performance on edge devices without needing PyTorch as a dependency in production.

## Verification Plan

### Synthetic Validation
- Evaluate distance error (RMSE of pixel distance between predicted corners and true corners) on a hold-out set of synthetic data.

### Real-world Validation
- Build a quick OpenCV webcam script that runs the ONNX model in real-time.
- Test with documents on difficult backgrounds (white paper on white desk, steep angles, holding paper in hand).
- Verify inference time is < 50ms per frame on CPU.
