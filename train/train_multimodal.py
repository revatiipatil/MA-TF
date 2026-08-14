#!/usr/bin/env python3
"""
Train the Multi-Modal Transformer (MA-TF) for chromatin accessibility prediction.

Inputs:
    - One-hot encoded DNA sequences
    - Histone modification features
    - Transcription factor binding features

Target:
    - ATAC-seq accessibility signal

ATAC-seq is used ONLY as the prediction target and is never passed
to the model as an input feature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_multimodal import MultiModalAccessibilityModel


# ============================================================
# Dataset
# ============================================================

class MultiModalDataset(Dataset):
    """
    Dataset containing DNA sequence, functional genomic features,
    and ATAC-seq target values.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        functional: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray | None = None,
    ):
        if indices is not None:
            sequences = sequences[indices]
            functional = functional[indices]
            targets = targets[indices]

        self.sequences = sequences.astype(np.float32)
        self.functional = functional.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "sequence": torch.from_numpy(self.sequences[idx]),
            "functional": torch.from_numpy(self.functional[idx]),
            "target": torch.tensor(
                self.targets[idx],
                dtype=torch.float32,
            ),
        }


# ============================================================
# Loss
# ============================================================

class CombinedLoss(nn.Module):
    """
    Combined MSE + Pearson correlation loss.

    The MSE term encourages accurate numerical predictions.
    The correlation term encourages preservation of the
    relationship between predicted and true accessibility.
    """

    def __init__(
        self,
        mse_weight: float = 0.5,
        correlation_weight: float = 0.5,
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        mse = self.mse_loss(
            predictions,
            targets,
        )

        pred_centered = (
            predictions - predictions.mean()
        )

        target_centered = (
            targets - targets.mean()
        )

        numerator = (
            pred_centered * target_centered
        ).sum()

        pred_std = (
            pred_centered ** 2
        ).sum()

        target_std = (
            target_centered ** 2
        ).sum()

        denominator = torch.sqrt(
            pred_std * target_std + 1e-8
        )

        correlation = numerator / denominator

        correlation = torch.clamp(
            correlation,
            -1.0,
            1.0,
        )

        correlation_loss = 1.0 - correlation

        return (
            self.mse_weight * mse
            + self.correlation_weight * correlation_loss
        )


# ============================================================
# Data loading
# ============================================================

def load_data(data_dir: str):
    """
    Load preprocessed multimodal data.

    Expected structure:

        data/processed/
        ├── sequences/
        │   └── encoded_sequences.npz
        │
        ├── functional/
        │   └── functional_features.npz
        │
        └── labels/
            ├── accessibility_scores.npy
            └── splits.npz

    The functional feature matrix MUST contain only histone
    and TF features. ATAC must not be included.
    """

    data_dir = Path(data_dir)

    print("Loading data...")

    # --------------------------------------------------------
    # DNA sequences
    # --------------------------------------------------------

    sequence_path = (
        data_dir
        / "sequences"
        / "encoded_sequences.npz"
    )

    sequence_data = np.load(
        sequence_path,
        allow_pickle=True,
    )

    sequences = sequence_data["X"]

    print(
        f"  Sequences: {sequences.shape}"
    )

    # --------------------------------------------------------
    # Functional genomic features
    # --------------------------------------------------------

    functional_path = (
        data_dir
        / "functional"
        / "functional_features.npz"
    )

    functional_data = np.load(
        functional_path,
        allow_pickle=True,
    )

    functional = functional_data["X"]

    print(
        f"  Functional features: {functional.shape}"
    )

    # --------------------------------------------------------
    # ATAC target
    # --------------------------------------------------------

    target_path = (
        data_dir
        / "labels"
        / "accessibility_scores.npy"
    )

    targets = np.load(target_path)

    print(
        f"  ATAC targets: {targets.shape}"
    )

    # --------------------------------------------------------
    # Train/validation/test splits
    # --------------------------------------------------------

    split_path = (
        data_dir
        / "labels"
        / "splits.npz"
    )

    splits = np.load(
        split_path,
        allow_pickle=True,
    )

    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]

    print(
        f"  Train: {len(train_indices)}"
    )

    print(
        f"  Validation: {len(val_indices)}"
    )

    print(
        f"  Test: {len(test_indices)}"
    )

    return (
        sequences,
        functional,
        targets,
        train_indices,
        val_indices,
        test_indices,
    )


# ============================================================
# Training
# ============================================================

def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(
        dataloader,
        desc="Training",
    ):

        sequence = batch["sequence"].to(device)
        functional = batch["functional"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        prediction = model(
            sequence,
            functional,
        )

        loss = criterion(
            prediction,
            target,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# Validation
# ============================================================

def evaluate_loss(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    n_batches = 0

    predictions = []
    targets = []

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc="Validation",
        ):

            sequence = batch["sequence"].to(device)
            functional = batch["functional"].to(device)
            target = batch["target"].to(device)

            prediction = model(
                sequence,
                functional,
            )

            loss = criterion(
                prediction,
                target,
            )

            total_loss += loss.item()
            n_batches += 1

            predictions.append(
                prediction.cpu().numpy()
            )

            targets.append(
                target.cpu().numpy()
            )

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    correlation = np.corrcoef(
        predictions,
        targets,
    )[0, 1]

    return (
        total_loss / max(n_batches, 1),
        correlation,
    )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_correlation,
    checkpoint_dir,
):
    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": float(val_loss),
        "val_correlation": float(val_correlation),
    }

    epoch_path = (
        checkpoint_dir
        / f"checkpoint_epoch_{epoch}.pt"
    )

    latest_path = (
        checkpoint_dir
        / "checkpoint_latest.pt"
    )

    torch.save(
        checkpoint,
        epoch_path,
    )

    torch.save(
        checkpoint,
        latest_path,
    )

    print(
        f"  Saved checkpoint: {epoch_path}"
    )


# ============================================================
# Main training function
# ============================================================

def main(args):

    print("=" * 70)
    print("MA-TF MULTI-MODAL TRAINING")
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

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: {torch.version.cuda}"
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
    ) = load_data(args.data_dir)

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = MultiModalDataset(
        sequences,
        functional,
        targets,
        train_indices,
    )

    val_dataset = MultiModalDataset(
        sequences,
        functional,
        targets,
        val_indices,
    )

    test_dataset = MultiModalDataset(
        sequences,
        functional,
        targets,
        test_indices,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating MA-TF model...")

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

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = CombinedLoss(
        mse_weight=args.mse_weight,
        correlation_weight=args.correlation_weight,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    train_losses = []
    val_losses = []
    val_correlations = []

    best_val_loss = float("inf")
    patience_counter = 0

    print(
        f"\nTraining for {args.epochs} epochs..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_correlation = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_loss)

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        val_correlations.append(
            val_correlation
        )

        print(
            f"Train Loss: {train_loss:.6f}"
        )

        print(
            f"Val Loss: {val_loss:.6f}"
        )

        print(
            f"Val Pearson: {val_correlation:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_correlation=val_correlation,
                checkpoint_dir=args.checkpoint_dir,
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{args.patience})"
            )

        if patience_counter >= args.patience:

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print("\nEvaluating on test set...")

    test_loss, test_correlation = evaluate_loss(
        model,
        test_loader,
        criterion,
        device,
    )

    print(
        f"Test Loss: {test_loss:.6f}"
    )

    print(
        f"Test Pearson: {test_correlation:.4f}"
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_correlations": val_correlations,
        "best_val_loss": float(best_val_loss),
        "final_test_loss": float(test_loss),
        "final_test_pearson": float(
            test_correlation
        ),
        "epochs_trained": len(train_losses),
    }

    history_path = (
        output_dir
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print(
        f"\nTraining history saved to: "
        f"{history_path}"
    )

    print("\nTraining complete.")


# ============================================================
# Command-line arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train MA-TF multimodal transformer"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
    )

    parser.add_argument(
        "--feature_map",
        type=str,
        required=True,
        help="Path to JSON feature-map file",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--mse_weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--correlation_weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/checkpoints",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/training",
    )

    args = parser.parse_args()

    # Load feature map
    with open(
        args.feature_map,
        "r",
        encoding="utf-8",
    ) as f:

        args.feature_map = json.load(f)

    main(args)