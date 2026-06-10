# Luggage Threat Detection

A computer vision pipeline for automated threat screening in X-ray luggage imagery.

## What The Current Code Does

The project is organized into two tasks:

1. Segmentation of threat regions from luggage scans
2. Binary classification of segmented results as `safe` or `threat`

Current runtime behavior in `processing.py`:

- By default, only `classify_image()` is executed when running `python main.py`.
- `segment_image()` exists and can be enabled manually in `Processing.__init__`.

## Project Structure

- `main.py`: sets TensorFlow environment flags and starts `Processing()`.
- `processing.py`: pipeline orchestration (segmentation + classification).
- `constants.py`: loads `.env` and exposes path constants.
- `config/classification_model.py`: CNN training/inference for binary classification.
- `config/segementation_model.py`: U-Net-like segmentation model.
- `final_output.json`: classification output grouped by predicted class.

## Environment Configuration

Create a `.env` file in the project root and define:

- `TRAIN`: root directory for segmentation training images.
- `TRAIN_ANNOTATION`: root directory for segmentation masks.
- `TEST`: root directory of test images used by segmentation inference.
- `TEST_ANNOTATION`: currently loaded but not used in pipeline logic.
- `RESULT`: output root directory for segmentation artifacts.

Expected segmentation subfolders under `TRAIN` and `TRAIN_ANNOTATION`:

- `GUN`
- `knife`
- `shuriken`

## Pipeline Details

### 1. Segmentation (`SegmentationModel`)

- Architecture: U-Net style encoder-decoder with skip connections.
- Input size: `512 x 512 x 3`.
- Loss: BCE + Dice loss.
- Metrics: Dice coefficient, IoU, accuracy.
- Saved model path: `model/segmentation_model.keras` (fallback load from `.h5` is supported).

When `segment_image()` runs:

- If no segmentation model exists (`.keras` or `.h5`), training starts.
- Test images are loaded from `TEST` recursively.
- For each image:
  - Binary mask is predicted.
  - Overlay image is generated.
  - Files are saved to:
    - `<RESULT>/result_annotation/<name>.png` (binary mask)
    - `<RESULT>/segment_images/<name>.png` (overlay)

### 2. Classification (`Classification_Model`)

- Architecture: custom CNN with Conv-BN blocks and dense head.
- Input size: `512 x 512 x 3`.
- Classes: `safe`, `threat`.
- Saved model path: `model/classification_model.h5`.

Training behavior:

- If model file does not exist, training is run using `TRAIN_ANNOTATION` as input root.
- The `safe` class is auto-generated if missing by creating synthetic images under `<TRAIN_ANNOTATION>/safe`.

Inference behavior:

- Predicts on `<RESULT>/result_annotation` using `image_dataset_from_directory`.
- Writes grouped results to `final_output.json`.

`final_output.json` format:

```json
{
  "threat": [
    {
      "file_name": "example.png",
      "predicted_class": "threat",
      "confidence": "99.12%"
    }
  ],
  "safe": []
}
```

## Installation

### Windows / Standard

```bash
pip install -r requirements.txt
```

### Linux (CUDA-ready variant in repo)

```bash
pip install -r requirements-linux.txt
```

## Run

```bash
python main.py
```

## GPU Reliability (Linux / WSL Recommended)

For more reliable GPU utilization, run training/inference on native Linux or WSL2 (Ubuntu) instead of Windows-native Python.

- Use `requirements-linux.txt` for GPU-oriented setup.
- In WSL2, install NVIDIA drivers on Windows host + CUDA support for WSL.
- Keep the project and dataset inside the Linux filesystem (for example under `~/`) for better I/O performance.

Quick TensorFlow GPU check:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If the output list is non-empty, TensorFlow can see your GPU.

## Important Notes

- Default entry flow currently runs classification only.
- If segmentation is skipped, `<RESULT>/result_annotation` must already contain images for classification inference.
- Segmentation and classification are currently binary threat detectors, not per-threat-type classifiers.

## References

- [Project Statement](https://drive.google.com/file/d/1bYLJDkE4hu52kihmjm95CCDjCUUXPTVl/view?usp=sharing)
- [Dataset](https://drive.google.com/drive/folders/1eOoN5LSE9OEyWFfA7ntPfZTOQjvdpRkB?usp=sharing)
- [Results Report](https://drive.google.com/file/d/1CPeFO8LYqyjLG0fbqAihXEbUZlZrKtz2/view)
