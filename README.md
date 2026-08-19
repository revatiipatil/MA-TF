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

## Data

The required genomic datasets can be obtained from the ENCODE Project.

Detailed instructions for downloading the required data and reproducing the preprocessing pipeline are provided in:

**[`data/readme.md`](data/readme.md)**

Small processed files required for reproducibility are included in the repository where appropriate. Large genomic files and generated arrays are excluded from Git.

## Installation

```bash
git clone <repository-url>
cd MA-TF

pip install -r requirements.txt