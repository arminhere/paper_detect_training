# Document Corner Detection (DocCornerNet)

This project contains a complete, end-to-end pipeline for training and deploying an AI-based document corner detection model. It uses a lightweight PyTorch model (MobileNetV3-Small backbone) to predict the 4 corners of a document in an image, which is then used to perform a perspective transformation (warping) to create a clean, flat scan.

The repository includes scripts to generate synthetic training data, train the model, export it to ONNX and TensorFlow Lite (TFLite) for mobile/edge use, and an interactive Tkinter GUI application to test the model.

## Prerequisites

Before starting, ensure you have Python 3.12 (or compatible) installed. Install the dependencies by running:

```bash
pip install -r requirements.txt
```

This will install PyTorch, OpenCV, TensorFlow, ONNX, and other required packages.

## Pipeline Walkthrough

Follow these steps to train the model from scratch and run the application.

### Step 1: Generating Synthetic Data

The model is trained entirely on synthetic data. The generation script automatically downloads random background images and overlays synthesized "documents" (complete with random text and lines) onto them with heavy perspective distortion.

To generate the dataset, run:

```bash
python generate_data.py
```

**What it does:**
- Downloads 100 random background images from `picsum.photos` into the `backgrounds/` folder.
- Generates 1,000 synthetic training images into the `dataset/images/` folder.
- Creates a `dataset/labels.json` file containing the precise coordinates (Top-Left, Top-Right, Bottom-Right, Bottom-Left) for each generated image.

*(Note: You can edit `generate_data.py` to increase the `num_samples` for better accuracy during training).*

### Step 2: Training the PyTorch Model

Once the data is generated, you can train the Document Corner Detection model. The model predicts 4 heatmaps, and during training, we minimize the Mean Squared Error (MSE) against ideal Gaussian blobs placed at the true corners.

To train the model, run:

```bash
python train.py
```

**What it does:**
- Loads the MobileNetV3-Small model and your synthetic dataset.
- Runs training for 15 epochs (configurable).
- Evaluates against a validation split and automatically saves the best performing weights to `weights/best_model.pth`.

### Step 3: Exporting to ONNX

To make the model portable for edge devices and other runtimes, the PyTorch model needs to be exported to the ONNX format.

Run the export script:

```bash
python export.py
```

**What it does:**
- Loads the trained `weights/best_model.pth` (or an untrained structure if weights are missing).
- Traces the model graph with a dummy `1x3x256x256` input.
- Exports the model as `doc_corner_net.onnx`.

### Step 4: Converting ONNX to TensorFlow Lite (TFLite)

For mobile and embedded deployments, TensorFlow Lite is preferred. We convert the ONNX model directly to TFLite using the `onnx2tf` library.

Run the conversion script:

```bash
python export_tflite.py
```

**What it does:**
- Runs the `onnx2tf` CLI tool under the hood to convert `doc_corner_net.onnx` into a TensorFlow saved model and then into a float32 `.tflite` model.
- Saves the output as `doc_corner_net_float32.tflite` in your root directory.

### Step 5: Testing with the Desktop App

You can test your newly trained `.tflite` model using the interactive desktop scanner application. The app allows you to open an image, automatically predicts the document corners using the AI model, and lets you drag handles to refine the warp.

Run the app:

```bash
python document_warp.py
```

*(You can also pass an image path directly: `python document_warp.py path/to/image.jpg`)*

**Features:**
- AI automatic paper detection upon image load (using `doc_corner_net_float32.tflite`).
- Supports hardware acceleration via the TFLite GPU delegate if available.
- Interactive perspective correction.
- Black & white document scan enhancement mode.

## Customizing the Model (Sim-to-Real fine-tuning)

If you find that the synthetic dataset doesn't perfectly match your real-world use case (the "sim-to-real" gap), you can fine-tune the model:

1. Use the `document_warp.py` app to visually verify model predictions on real photos.
2. If the corners are slightly off, record the correct coordinates.
3. Add these real photos and their corrected coordinates to your `dataset/labels.json` and retrain using `train.py`.
