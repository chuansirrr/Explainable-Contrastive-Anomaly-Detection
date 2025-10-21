# Explainable-Contrastive-Anomaly-Detection

Contrastive pretraining + explainable anomaly detection (tabular & time-series) with gradient saliency and permutation importance.

This repository implements **contrastive representation learning** for anomaly detection and couples it with **explainability** tools so you can understand *why* points are flagged as anomalous.

## Highlights
- **SimCLR-style contrastive pretraining** on unlabeled data (`NT-Xent` loss).
- **Detector heads**: Mahalanobis distance, kNN in embedding space, and one-class SVM.
- **Explainability**:
  - **Gradient Saliency** on the encoder to attribute anomaly scores to input features.
  - **Permutation Feature Importance (PFI)** for model-agnostic attribution of the anomaly score.
- **Synthetic data generators** for tabular and time-series with controllable anomaly rate.
- **Reproducible experiments**: YAML configs, deterministic seeds, CLI scripts, tests, CI, and Dockerfile.

## Quickstart

```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Pretrain encoder contrastively (unsupervised)
python scripts/pretrain_contrastive.py --config configs/tabular_small.yaml

# 3) Train anomaly detector head on embeddings
python scripts/train_detector.py --config configs/tabular_small.yaml --head mahalanobis

# 4) Evaluate
python scripts/evaluate.py --config configs/tabular_small.yaml --head mahalanobis

# 5) Explain flagged anomalies
python scripts/explain.py --config configs/tabular_small.yaml --head mahalanobis --method saliency
```

## Project Layout
```
.
├─ xadl/                        # Core library
│  ├─ data.py                   # Datasets & augmentations
│  ├─ losses.py                 # NT-Xent and utilities
│  ├─ utils.py                  # Seed, device, logging helpers
│  ├─ models/
│  │   ├─ encoder.py            # MLP & 1D-CNN encoders
│  │   └─ heads.py              # Mahalanobis, kNN, OneClassSVM
│  ├─ training/
│  │   ├─ contrastive_trainer.py
│  │   └─ detector_trainer.py
│  └─ explain/
│      └─ explain.py            # Saliency & permutation importance
├─ scripts/                     # CLI entry points
├─ configs/                     # YAML config files
├─ notebooks/                   # Demos
├─ tests/                       # Unit tests (pytest)
├─ .github/workflows/python-ci.yml
├─ requirements.txt
├─ pyproject.toml
├─ LICENSE
└─ CITATION.cff
```

## Datasets
This repo ships with **synthetic generators**; you can also drop CSVs into `data/` with a header row (features only, unlabeled). See `xadl/data.py` for expected shapes and `--dataset` CLI args.

## Citing
See `CITATION.cff`.
