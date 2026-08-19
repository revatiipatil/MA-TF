# MA-TF

## Modality-Aware Transformer for Chromatin Accessibility Prediction

MA-TF is a multimodal transformer framework for predicting continuous
chromatin accessibility from DNA sequence and functional genomic signals.

The model integrates:

- DNA sequence
- H3K27ac
- H3K4me3
- CTCF

ATAC-seq accessibility is used exclusively as the prediction target and
is not provided to the model as an input feature.

This separation prevents target leakage and ensures that the model predicts
chromatin accessibility from independent genomic modalities.

---

## Overview

The MA-TF pipeline consists of four main stages:

```text
ENCODE / hg38 data
        |
        v
Preprocessing
        |
        +--> DNA sequence
        |
        +--> H3K27ac
        +--> H3K4me3
        +--> CTCF
        |
        +--> ATAC-seq target
        |
        v
Dataset alignment and normalization
        |
        v
Multimodal Transformer
        |
        v
Continuous accessibility prediction
        |
        v
Evaluation and visualization