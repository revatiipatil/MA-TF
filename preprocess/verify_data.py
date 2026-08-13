#!/usr/bin/env python3
"""Verify that all data files are ready for training"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("Data Verification")
print("=" * 60)

# Check sequences
seq_path = Path("data/processed/sequences/encoded_sequences.npz")
if seq_path.exists():
    data = np.load(seq_path, allow_pickle=True)
    print(f"\n1. Sequences: OK")
    print(f"   File: {seq_path}")
    print(f"   Shape: {data['X'].shape}")
    print(f"   Total sequences: {data['X'].shape[0]}")
else:
    print(f"\n1. Sequences: MISSING")

# Check labels
label_path = Path("data/processed/labels/accessibility_scores.npy")
if label_path.exists():
    labels = np.load(label_path)
    print(f"\n2. Labels: OK")
    print(f"   File: {label_path}")
    print(f"   Shape: {labels.shape}")
    print(f"   Mean: {labels.mean():.4f}")
    print(f"   Std: {labels.std():.4f}")
    print(f"   Min: {labels.min():.4f}")
    print(f"   Max: {labels.max():.4f}")
    print(f"   NOTE: These are placeholder scores. Install pyBigWig for real scores!")
else:
    print(f"\n2. Labels: MISSING")

# Check splits
split_path = Path("data/processed/labels/splits.npz")
if split_path.exists():
    splits = np.load(split_path, allow_pickle=True)
    print(f"\n3. Train/Val/Test Splits: OK")
    print(f"   File: {split_path}")
    print(f"   Train: {len(splits['train'])} sequences ({len(splits['train'])/455*100:.1f}%)")
    print(f"   Val:   {len(splits['val'])} sequences ({len(splits['val'])/455*100:.1f}%)")
    print(f"   Test:  {len(splits['test'])} sequences ({len(splits['test'])/455*100:.1f}%)")
else:
    print(f"\n3. Splits: MISSING")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

all_ready = (
    seq_path.exists() and 
    label_path.exists() and 
    split_path.exists()
)

if all_ready:
    print("\nSUCCESS: All data files are ready for training!")
    print("\nNext steps:")
    print("1. Create model architecture (scripts/train/model_epibert.py)")
    print("2. Create training script (scripts/train/train_epibert.py)")
    print("3. Start training!")
else:
    print("\nWARNING: Some files are missing. Please check above.")

print("\n" + "=" * 60)

