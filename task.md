# Document Corner Detection Tasks

- `[ ]` **Phase 1: Synthetic Data Generation**
  - `[ ]` Prepare background images dataset.
  - `[ ]` Write script to generate fake foreground documents (`reportlab`, `faker`).
  - `[ ]` Implement data augmentation and generator pipeline (perspective transforms, blur, noise).
- `[ ]` **Phase 2: Model Architecture**
  - `[ ]` Implement MobileNetV3 backbone and heatmap prediction head.
- `[ ]` **Phase 3: Training Strategy**
  - `[ ]` Create PyTorch Dataset and DataLoader.
  - `[ ]` Implement MSE loss and training loop.
- `[ ]` **Phase 4: Inference Pipeline**
  - `[ ]` Implement preprocessing and heatmap-to-coordinate extraction.
  - `[ ]` Implement homography perspective correction.
- `[ ]` **Phase 5: Deployment**
  - `[ ]` Export model to ONNX format.
