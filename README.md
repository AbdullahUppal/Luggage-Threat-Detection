# Luggage Threat Detection

A computer vision system for automated detection of prohibited items in luggage for enhanced public safety at airports, malls, and cargo terminals.

## Project Overview

This project implements an automated system that:
1. **Classifies** baggage images as safe or threat
2. **Identifies** threat types (gun, knife, shuriken)
3. **Segments** threat object regions from baggage images

## Current Implementation

The codebase has been consolidated into a modular pipeline with the following structure:

### Core Components

- **`main.py`** - Entry point for the pipeline
- **`processing.py`** - Main processing orchestration
  - Segmentation model training and inference
  - Classification model training and inference
- **`constants.py`** - Configuration and path constants
- **`config/`** - Model configuration modules
  - `segmentation_model.py` - Segmentation model implementation
  - `classification_model.py` - Classification model implementation

### Dependencies

```
pandas==3.0.3
opencv-python==4.13.0.92
python-dotenv==1.2.2
keras==3.14.1
matplotlib==3.10.9
tensorflow==2.21.0
```

## Dataset

The system uses a structured dataset with:
- **Threat classes:** gun, knife, shuriken
- **Safe images:** non-threat samples
- **Structure:** Train/test splits with corresponding annotation masks
- **Details:** See [Dataset folder](https://drive.google.com/drive/folders/1eOoN5LSE9OEyWFfA7ntPfZTOQjvdpRkB?usp=sharing)

## Model Architecture

### Segmentation Pipeline
- Trains on annotated threat images
- Uses mask-based supervision for pixel-level segmentation
- Model saved to `model/segmentation_model.pt`

### Classification Pipeline
- Classifies images as safe or threat
- Supports multi-class threat categorization
- Model saved to `model/classification_model.pt`

## Evaluation Metrics

### Classification
- Overall accuracy
- Confusion matrix
- Per-class precision, recall, F1-score

### Segmentation
- Dice coefficient (F1 score)
- Intersection over Union (IoU)

## Configuration

Environment variables are loaded via `.env` file. Key paths configured in `constants.py`:

- `TRAIN_*` - Training data paths for each threat class
- `TEST_*` - Test data paths
- `*_ANNOTATION` - Segmentation mask paths

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Configure dataset paths in .env file
# Run the pipeline
python main.py
```

## Project Status

✅ **Completed:**
- Modular pipeline architecture
- Standardized preprocessing
- Reproducible train/test workflow
- Model training and evaluation for both tasks

🔄 **In Progress:**
- Fine-tuning model performance
- Comprehensive evaluation reporting
- Documentation and sample outputs

For detailed results and analysis, see the [full report](https://drive.google.com/file/d/1CPeFO8LYqyjLG0fbqAihXEbUZlZrKtz2/view).

## License

This project is for research and security screening purposes.

## References

- [Project Statement](https://drive.google.com/file/d/1bYLJDkE4hu52kihmjm95CCDjCUUXPTVl/view?usp=sharing)
- [Dataset](https://drive.google.com/drive/folders/1eOoN5LSE9OEyWFfA7ntPfZTOQjvdpRkB?usp=sharing)
- [Results Report](https://drive.google.com/file/d/1CPeFO8LYqyjLG0fbqAihXEbUZlZrKtz2/view)
