# Luggage Threat Detection

## Project Status
This repository is currently in development.

The codebase contains exploratory and legacy implementations, and the final unified training/inference pipeline is still being consolidated.

## Issue Statement
Detecting prohibited items in luggage is important for public safety in places such as airports, malls, and cargo terminals. Manual screening is slow, labor-intensive, and prone to human error due to fatigue and workload.

The project goal is to build an automated computer vision system that can:
1. Classify a baggage image as safe or threat.
2. If threat is present, identify the threat type.
3. Segment the threat object region from the baggage image.

### Complete statement document
https://drive.google.com/file/d/1bYLJDkE4hu52kihmjm95CCDjCUUXPTVl/view?usp=sharing



### Required Tasks
1. Classification
- Classify each input image as safe or threat.
- For threat images, target categories include: gun, knife, and shuriken.

2. Segmentation
- Perform pixel-level segmentation of the threat object in baggage images using provided masks.

### Dataset details
https://drive.google.com/drive/folders/1eOoN5LSE9OEyWFfA7ntPfZTOQjvdpRkB?usp=sharing

### Dataset Summary
- Data is split into train and test sets.
- Both sets contain class folders and annotation masks.
- Threat classes mentioned in the assignment: gun, knife, shuriken.
- Safe images are included as non-threat samples.

### Evaluation Metrics
Classification metrics:
- Overall accuracy
- Confusion matrix
- Dice score

Segmentation metric:
- Dice coefficient (F1 score)

## Current Development Direction
The active development target is to:
1. Refactor experimental scripts into a clean, modular pipeline.
2. Standardize preprocessing and label/mask loading.
3. Implement a reproducible train/test workflow for both classification and segmentation.
4. Add consistent evaluation reporting (accuracy, confusion matrix, Dice).
5. Finalize documentation and sample outputs.

## Planned Structure (Work in Progress)
- data loading and preprocessing
- classification model training and evaluation
- segmentation model training and evaluation
- inference scripts
- results and visualizations
- report-ready metrics tables

