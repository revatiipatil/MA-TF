#!/usr/bin/env python3
"""
Training script for EpiBERT model.

Trains the transformer model to predict chromatin accessibility from DNA sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from model_epibert import create_tiny_epibert


class CombinedLoss(nn.Module):
    """
    Combined loss function: MSE + (1 - Pearson correlation)
    This encourages the model to minimize prediction error AND maximize correlation.
    """
    def __init__(self, mse_weight=0.5, correlation_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, labels):
        # MSE component
        mse = self.mse_loss(predictions, labels)
        
        # Pearson correlation component (maximize correlation = minimize 1 - correlation)
        # Normalize predictions and labels
        pred_mean = predictions.mean()
        label_mean = labels.mean()
        pred_centered = predictions - pred_mean
        label_centered = labels - label_mean
        
        # Compute correlation
        numerator = (pred_centered * label_centered).sum()
        pred_std = (pred_centered ** 2).sum()
        label_std = (label_centered ** 2).sum()
        denominator = torch.sqrt(pred_std * label_std + 1e-8)  # Add epsilon for numerical stability
        
        # Compute correlation
        correlation = numerator / denominator
        
        # Correlation loss: 1 - correlation (we want to maximize correlation)
        # Clip correlation to valid range [-1, 1] for stability
        correlation = torch.clamp(correlation, -1.0, 1.0)
        correlation_loss = 1.0 - correlation
        
        # Combined loss
        total_loss = self.mse_weight * mse + self.correlation_weight * correlation_loss
        
        return total_loss


class ChromatinAccessibilityDataset(Dataset):
    """Dataset for chromatin accessibility prediction."""
    
    def __init__(self, sequences, labels, indices=None):
        """
        Args:
            sequences: numpy array of shape (N, seq_len, 4) - one-hot encoded sequences
            labels: numpy array of shape (N,) - accessibility scores
            indices: optional list of indices to subset the data
        """
        self.sequences = sequences
        self.labels = labels
        
        if indices is not None:
            self.sequences = self.sequences[indices]
            self.labels = self.labels[indices]
        
        # Convert to float32
        self.sequences = self.sequences.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'sequence': torch.from_numpy(self.sequences[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_data(data_dir="data/processed"):
    """Load sequences, labels, and splits."""
    data_dir = Path(data_dir)
    
    print("Loading data...")
    
    # Load sequences
    sequences_data = np.load(data_dir / "sequences" / "encoded_sequences.npz", allow_pickle=True)
    sequences = sequences_data['X']  # (N, seq_len, 4)
    print(f"  Sequences: {sequences.shape}")
    
    # Load labels
    labels = np.load(data_dir / "labels" / "accessibility_scores.npy")
    print(f"  Labels: {labels.shape}")
    
    # Load splits
    splits = np.load(data_dir / "labels" / "splits.npz", allow_pickle=True)
    train_indices = splits['train']
    val_indices = splits['val']
    test_indices = splits['test']
    
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return sequences, labels, train_indices, val_indices, test_indices


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(sequences)
        
        # Use combined loss (MSE + Correlation)
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / n_batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Calculate correlation
    correlation = np.corrcoef(all_predictions, all_labels)[0, 1]
    
    return avg_loss, correlation


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Also save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)


def plot_training_curves(train_losses, val_losses, val_correlations=None, output_dir=None):
    """Plot training curves with loss and correlation."""
    output_dir = Path(output_dir) if output_dir else Path("results/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Correlation curve
    if val_correlations:
        axes[1].plot(val_correlations, label='Val Correlation', marker='^', color='green', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero correlation')
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Pearson Correlation', fontweight='bold')
        axes[1].set_title('Validation Correlation Over Time', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved training curves: {plot_path}")
    plt.close()


def main(args):
    print("=" * 60)
    print("EpiBERT Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  Using CPU (GPU not available)")
    
    # Load data
    sequences, labels, train_indices, val_indices, test_indices = load_data(args.data_dir)
    
    # Create datasets
    train_dataset = ChromatinAccessibilityDataset(sequences, labels, train_indices)
    val_dataset = ChromatinAccessibilityDataset(sequences, labels, val_indices)
    test_dataset = ChromatinAccessibilityDataset(sequences, labels, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print(f"\nCreating model...")
    model = create_tiny_epibert(
        seq_len=sequences.shape[1],
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss function - Use combined MSE + Correlation loss to prevent collapse
    print(f"\nLoss function: Combined MSE + Correlation Loss")
    print(f"  MSE weight: {args.mse_weight:.2f}")
    print(f"  Correlation weight: {args.correlation_weight:.2f}")
    criterion = CombinedLoss(mse_weight=args.mse_weight, correlation_weight=args.correlation_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    def warmup_lambda(epoch):
        if epoch <= 3:
            return (epoch + 1) / 4  # Warmup for first 3 epochs
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Reduce LR on plateau after warmup
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    val_correlations = []
    best_val_loss = float('inf')
    best_val_correlation = -float('inf')
    patience_counter_loss = 0
    patience_counter_corr = 0
    early_stop_patience = args.early_stop_patience
    
    print(f"\nEarly stopping patience: {early_stop_patience} epochs")
    print(f"Early stopping based on: {'Correlation' if args.early_stop_on_correlation else 'Loss'}")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_correlation = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_correlations.append(val_correlation)
        
        # Learning rate scheduling - use correlation if available, else loss
        if epoch <= 3:
            warmup_scheduler.step()  # Warmup phase
        else:
            # Use negative correlation for plateau scheduler (we want to maximize correlation)
            if args.lr_schedule_on_correlation:
                plateau_scheduler.step(-val_correlation)  # Negative because we want to maximize
            else:
                plateau_scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val Correlation: {val_correlation:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best checkpoint based on correlation (more important than loss)
        improved = False
        if args.early_stop_on_correlation:
            # Early stop and save based on correlation
            if val_correlation > best_val_correlation:
                best_val_correlation = val_correlation
                best_val_loss = val_loss
                patience_counter_corr = 0
                improved = True
                save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
                print(f"  ✓ New best model! (val_correlation: {val_correlation:.4f})")
            else:
                patience_counter_corr += 1
                print(f"  No correlation improvement ({patience_counter_corr}/{early_stop_patience})")
            
            # Also track loss improvement for monitoring
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        else:
            # Early stop and save based on loss (original behavior)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_correlation = val_correlation
                patience_counter_loss = 0
                improved = True
                save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
                print(f"  ✓ New best model! (val_loss: {val_loss:.6f})")
            else:
                patience_counter_loss += 1
                print(f"  No improvement ({patience_counter_loss}/{early_stop_patience})")
        
        # Early stopping
        if args.early_stop_on_correlation:
            if patience_counter_corr >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (correlation not improving)")
                print(f"Best validation correlation: {best_val_correlation:.4f}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break
        else:
            if patience_counter_loss >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (loss not improving)")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"Best validation correlation: {best_val_correlation:.4f}")
                break
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
    
    # Final evaluation on test set
    print(f"\nEvaluating on test set...")
    test_loss, test_correlation = validate(model, test_loader, criterion, device)
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Correlation: {test_correlation:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_correlations, args.output_dir)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_correlations': val_correlations,
        'best_val_loss': float(best_val_loss),
        'best_val_correlation': float(best_val_correlation),
        'best_epoch': int(np.argmin(val_losses) + 1),
        'final_test_loss': float(test_loss),
        'final_test_correlation': float(test_correlation),
        'total_epochs_trained': len(train_losses)
    }
    
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history: {history_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EpiBERT model")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Directory containing processed data")
    parser.add_argument("--checkpoint_dir", type=str, default="models/pretrained",
                       help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, default="results/training",
                       help="Directory to save training outputs")
    
    # Model hyperparameters
    parser.add_argument("--embed_dim", type=int, default=256,
                       help="Embedding dimension (default: 256)")
    parser.add_argument("--n_layers", type=int, default=4,
                       help="Number of transformer layers (default: 4)")
    parser.add_argument("--n_heads", type=int, default=8,
                       help="Number of attention heads (default: 8)")
    parser.add_argument("--dim_feedforward", type=int, default=1024,
                       help="Feedforward dimension (default: 1024)")
    parser.add_argument("--dropout", type=float, default=0.25,
                       help="Dropout rate (default: 0.25, increased to prevent collapse)")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4, lower for stability)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay (default: 1e-4, increased to prevent collapse)")
    parser.add_argument("--mse_weight", type=float, default=0.5,
                       help="Weight for MSE component in combined loss (default: 0.5)")
    parser.add_argument("--correlation_weight", type=float, default=0.5,
                       help="Weight for correlation component in combined loss (default: 0.5)")
    parser.add_argument("--early_stop_on_correlation", action="store_true",
                       help="Early stop based on correlation improvement instead of loss")
    parser.add_argument("--lr_schedule_on_correlation", action="store_true",
                       help="Schedule learning rate based on correlation instead of loss")
    parser.add_argument("--save_every", type=int, default=5,
                       help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--early_stop_patience", type=int, default=15,
                       help="Early stopping patience (default: 15 epochs)")
    
    args = parser.parse_args()
    main(args)

