# MA-TF

## Modality-Aware Transformer for Chromatin Accessibility Prediction

MA-TF is a multimodal transformer framework for predicting continuous chromatin accessibility from DNA sequence and functional genomic signals.

The model integrates:

- DNA sequence
- H3K27ac
- H3K4me3
- CTCF

ATAC-seq accessibility is used exclusively as the prediction target and is not provided as an input feature.

## Repository

The repository contains scripts for:

- preprocessing genomic data
- generating sequence and functional features
- training the multimodal transformer
- evaluating the trained model
- training a sequence-only baseline

## File Structure

```text
MA-TF/
├── preprocess/              # Data preprocessing and feature generation
├── train/                   # Model training and evaluation
├── data/
│   ├── raw/                 # Raw genomic data (excluded from Git)
│   ├── processed/           # Processed data
│   └── readme.md            # Data download and preprocessing instructions
├── requirements.txt
└── README.md
```

## Data

The required genomic datasets can be obtained from the ENCODE Project.

Instructions for downloading the required data and reproducing the preprocessing pipeline are provided in:

**[`data/readme.md`](data/readme.md)**

Large genomic files and generated arrays are excluded from Git. Small processed files required for reproducibility are included where appropriate.

## Installation

```bash
git clone <repository-url>
cd MA-TF
pip install -r requirements.txt
```

## Training

After completing the data preparation described in [`data/readme.md`](data/readme.md), train the multimodal model with:

```bash
python train/train_multimodal.py
```

The training script uses the processed data under `data/processed/` and saves model checkpoints to the configured output directory.

## Evaluation

Evaluate a trained MA-TF checkpoint with:

```bash
python train/evaluate_multimodal.py --checkpoint <path-to-checkpoint>
```

## Sequence-Only Baseline

The sequence-only baseline can be trained with:

```bash
python train/seq-only/train_epibert.py
```

## Reproducibility

The preprocessing pipeline, model implementation, training, evaluation, and baseline scripts are provided in this repository.

See [`data/readme.md`](data/readme.md) for the complete data preparation procedure.

## License

See `LICENSE` for license information.