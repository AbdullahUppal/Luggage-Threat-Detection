# Luggage Threat Detection

Luggage Threat Detection is a PyTorch-based computer vision pipeline for automated screening of baggage X-ray style images. The system supports threat segmentation and threat-type classification to assist safety workflows in airports, malls, and cargo terminals.

## Project Overview

The implemented pipeline provides:

1. Pixel-level segmentation of threat objects from baggage images
2. Multi-class threat classification (gun, knife, shuriken, safe)
3. Model training, loading, and inference in a modular structure

## Implementation Summary

The repository is organized into reusable modules:

- `main.py`
  - Entry point that creates the processing pipeline object
- `utils/processing.py`
  - Orchestrates segmentation and classification steps
  - Runs segmentation inference over test images and stores generated masks
- `constants.py`
  - Loads runtime paths from `.env`
- `config/segementation_model.py`
  - PyTorch U-Net style segmentation model
  - Dataset pairing, training loop, model save/load, mask inference
- `config/classification_model.py`
  - PyTorch CNN classifier
  - Dataset preparation, train/validation split, training loop, prediction export

## Dependencies

Install dependencies from `requirements.txt`:

```bash
pandas==2.2.3
opencv-python==4.10.0.84
python-dotenv==1.0.1
matplotlib==3.8.4
torch==2.5.1+cu121
torchvision==0.20.1+cu121
--extra-index-url https://download.pytorch.org/whl/cu121
```

## Dataset Structure

Expected dataset layout includes train/test image folders and annotation masks.

- Threat classes: `GUN`, `knife`, `shuriken`
- Safe class: `safe`
- Training masks: threat-class masks aligned by filename

Referenced dataset and report:

- Dataset: [Google Drive folder](https://drive.google.com/drive/folders/1eOoN5LSE9OEyWFfA7ntPfZTOQjvdpRkB?usp=sharing)
- Full report: [Results report](https://drive.google.com/file/d/1CPeFO8LYqyjLG0fbqAihXEbUZlZrKtz2/view)

## Model Artifacts

Generated model files:

- Segmentation model: `model/segmentation_model.pt`
- Classification model: `model/classification_model.pt`

Note: `model/` is ignored by `.gitignore` and is intended for local runtime artifacts.

Generated output files:

- Segmented masks: `DIP Data Upload/result/`
- Classification summary: `classification_result.json`

Note: `DIP Data Upload/` is ignored by `.gitignore`, so generated images/results remain local.

## Configuration

Create a `.env` file in the project root and define:

```env
TRAIN=path_to_training_images
TRAIN_ANNOTATION=path_to_training_masks
TEST=path_to_test_images
```

These values are loaded by `constants.py` at runtime.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure dataset paths in `.env`.

3. Run the pipeline:

```bash
python main.py
```

### What Happens On Run

- Segmentation model is loaded from disk if available; otherwise it is trained and then saved.
- Test images are processed and mask outputs are written to `DIP Data Upload/result/`.
- Classification module is available in the processing pipeline for threat-label prediction and result export.

## Evaluation Signals

Current training loops report:

- Segmentation: train and validation BCE loss per epoch
- Classification: train/validation loss and accuracy per epoch

## Project Status

Completed:

- End-to-end modular PyTorch implementation
- Segmentation and classification model training pipelines
- Model persistence and reload flow
- Inference output generation for segmentation and classification
- Environment-driven path configuration

## License

This project is intended for research and security screening applications.

## References

- [Project Statement](https://drive.google.com/file/d/1bYLJDkE4hu52kihmjm95CCDjCUUXPTVl/view?usp=sharing)
- [Dataset](https://drive.google.com/drive/folders/1eOoN5LSE9OEyWFfA7ntPfZTOQjvdpRkB?usp=sharing)
- [Results Report](https://drive.google.com/file/d/1CPeFO8LYqyjLG0fbqAihXEbUZlZrKtz2/view)
