# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cargo is a deep learning image classification system that classifies geometric shapes (lines, parabolas, sine waves) using transfer learning with MobileNetV3-Small. It trains on synthetic data and calibrates confidence thresholds on real data.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Generate Synthetic Training Data
```bash
python scripts/generate_synthetic_lines.py
python scripts/generate_synthetic_parabolas.py
python scripts/generate_synthetic_sines.py
```

### Train and Evaluate
```bash
python main.py --config configs/default.yaml
```

### Run Evaluation Only
```bash
python -m src.evaluate --config configs/default.yaml --model outputs/model.pt
```

## Architecture

- **Entry point**: `main.py` loads YAML config, runs `train()` then `evaluate()`
- **Model** (`src/model.py`): `ShapeClassifier` is a `pl.LightningModule` wrapping MobileNetV3-Small with a custom classifier head (Linear→Hardswish→Dropout→Linear). Uses CrossEntropyLoss, Adam optimizer, cosine annealing scheduler.
- **Data** (`src/dataset.py`): `ShapeDataModule` is a `pl.LightningDataModule`. Images are loaded from class-named subdirectories under `data/synthetic/`. Training augmentations include random flip, rotation, ColorJitter, GaussianBlur. All images normalized with ImageNet stats.
- **Training** (`src/train.py`): PyTorch Lightning Trainer with ModelCheckpoint (best val_loss) and EarlyStopping (patience=5). Seed 42 for reproducibility.
- **Evaluation** (`src/evaluate.py`): Runs test set, generates confusion matrix PNGs and classification report text files, then triggers calibration.
- **Calibration** (`src/calibrate.py`): Computes per-class confidence thresholds by maximizing F1 score using precision-recall curves on real data. Saves thresholds to `outputs/thresholds.json`.
- **Inference** (`src/inference.py`): Loads saved model and thresholds for single-image prediction.

## Key Conventions

- All hyperparameters are config-driven via `configs/default.yaml`
- Three shape classes: `line`, `parabola`, `sine`
- Synthetic data lives in `data/synthetic/{class}/`, real data in `data/real_data/{class}/`
- Model artifacts saved to `outputs/` (checkpoints, model.pt, thresholds.json, confusion matrices, test reports)
