#!/usr/bin/env python3
"""
Evaluate a trained MA-TF model.

Inputs:
    - One-hot encoded DNA sequence
    - Histone modification features
    - TF binding features

Target:
    - ATAC-seq accessibility

ATAC-seq is used only as the evaluation target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_multimodal import MultiModalAccessibilityModel
from train_multimodal import (
    MultiModalDataset,
    load_data,
)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
):
    """
    Calculate regression and threshold-based metrics.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = np.mean(
        (predictions - targets) ** 2
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(mse)

    # --------------------------------------------------------
    # Pearson correlation
    # --------------------------------------------------------

    pearson = np.corrcoef(
        predictions,
        targets,
    )[0, 1]

    # --------------------------------------------------------
    # Spearman correlation
    # --------------------------------------------------------

    try:

        from scipy.stats import spearmanr

        spearman = spearmanr(
            predictions,
            targets,
        ).statistic

    except ImportError:

        spearman = float("nan")

    # --------------------------------------------------------
    # R-squared
    # --------------------------------------------------------

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - targets.mean()) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else float("nan")
    )

    # --------------------------------------------------------
    # Binary classification metrics
    #
    # Threshold = median target.
    # This is provided as an optional auxiliary evaluation.
    # --------------------------------------------------------

    threshold = np.median(targets)

    true_binary = (
        targets >= threshold
    ).astype(int)

    pred_binary = (
        predictions >= threshold
    ).astype(int)

    tp = np.sum(
        (pred_binary == 1)
        & (true_binary == 1)
    )

    tn = np.sum(
        (pred_binary == 0)
        & (true_binary == 0)
    )

    fp = np.sum(
        (pred_binary == 1)
        & (true_binary == 0)
    )

    fn = np.sum(
        (pred_binary == 0)
        & (true_binary == 1)
    )

    accuracy = (
        (tp + tn)
        / max(tp + tn + fp + fn, 1)
    )

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-8)
    )

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# Prediction
# ============================================================

def generate_predictions(
    model,
    dataloader,
    device,
):
    """
    Generate predictions for the evaluation dataset.
    """

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in dataloader:

            sequence = batch[
                "sequence"
            ].to(device)

            functional = batch[
                "functional"
            ].to(device)

            target = batch[
                "target"
            ].to(device)

            prediction = model(
                sequence,
                functional,
            )

            predictions.append(
                prediction.cpu().numpy()
            )

            targets.append(
                target.cpu().numpy()
            )

    predictions = np.concatenate(
        predictions
    )

    targets = np.concatenate(
        targets
    )

    return predictions, targets


# ============================================================
# Main
# ============================================================

def main(args):

    print("=" * 70)
    print("MA-TF MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        sequences,
        functional,
        targets,
        train_indices,
        val_indices,
        test_indices,
    ) = load_data(
        args.data_dir
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = MultiModalDataset(
        sequences,
        functional,
        targets,
        test_indices,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = MultiModalAccessibilityModel(
        seq_len=sequences.shape[1],
        feature_map=args.feature_map,
        seq_embed_dim=args.seq_embed_dim,
        fusion_embed_dim=args.fusion_embed_dim,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        dropout=args.dropout,
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint_path = Path(
        args.checkpoint
    )

    print(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions, true_targets = (
        generate_predictions(
            model,
            test_loader,
            device,
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        predictions,
        true_targets,
    )

    print("\nEvaluation results")
    print("-" * 40)

    for name, value in metrics.items():

        print(
            f"{name:15s}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_dir
        / "test_predictions.npz"
    )

    np.savez_compressed(
        predictions_path,
        predictions=predictions,
        targets=true_targets,
    )

    print(
        f"\nPredictions saved to: "
        f"{predictions_path}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = (
        output_dir
        / "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        f"Metrics saved to: "
        f"{metrics_path}"
    )

    print("\nEvaluation complete.")


# ============================================================
# Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate trained MA-TF model"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
    )

    parser.add_argument(
        "--feature_map",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
    )

    # Model parameters must match training.

    parser.add_argument(
        "--seq_embed_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--fusion_embed_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--fusion_layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--fusion_heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    with open(
        args.feature_map,
        "r",
        encoding="utf-8",
    ) as f:

        args.feature_map = json.load(f)

    main(args)