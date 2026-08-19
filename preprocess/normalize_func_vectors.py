#!/usr/bin/env python3
"""
Normalize functional genomic features for MA-TF.

Input:
    data/processed/functional/functional_features.npz

The input NPZ contains:
    X           : functional features, shape (N, 48)
    valid_mask  : validity mask, shape (N, 48)
    feature_map : JSON metadata

Normalization:
    Features are standardized independently for each bin:
        normalized = (x - mean) / std

    Normalization statistics are computed using only valid values.

Output:
    data/processed/functional/functional_features_normalized.npz

The output contains:
    X
    valid_mask
    feature_map
    normalization_stats
"""

from pathlib import Path
import json

import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "functional"
    / "functional_features.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "functional"
    / "functional_features_normalized.npz"
)

EPS = 1e-6


# ============================================================
# Normalization
# ============================================================

def normalize_features(
    features: np.ndarray,
    valid_mask: np.ndarray,
    feature_map: dict,
):
    """
    Standardize each functional feature independently.

    Invalid/missing values are excluded when computing mean/std.
    """

    normalized = features.astype(np.float32).copy()

    tracks = feature_map["tracks"]
    n_bins = feature_map["n_bins"]

    statistics = {}

    for track_idx, track_name in enumerate(tracks):

        start = track_idx * n_bins
        end = start + n_bins

        statistics[track_name] = {
            "mean": [],
            "std": [],
        }

        for feature_idx in range(start, end):

            valid = valid_mask[:, feature_idx].astype(bool)

            values = features[valid, feature_idx]

            if len(values) == 0:
                mean = 0.0
                std = 1.0

            else:
                mean = float(np.mean(values))
                std = float(np.std(values))

                if std < EPS:
                    std = 1.0

            normalized[:, feature_idx] = (
                features[:, feature_idx] - mean
            ) / std

            statistics[track_name]["mean"].append(mean)
            statistics[track_name]["std"].append(std)

    return normalized, statistics


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MA-TF Functional Feature Normalization")
    print("=" * 60)

    print("\nInput:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Load functional features
    # --------------------------------------------------------

    data = np.load(
        INPUT_FILE,
        allow_pickle=True,
    )

    features = data["X"]
    valid_mask = data["valid_mask"]

    feature_map_raw = data["feature_map"]

    if isinstance(feature_map_raw, np.ndarray):
        feature_map_raw = feature_map_raw.item()

    feature_map = json.loads(feature_map_raw)

    print(f"\nFeatures:   {features.shape}")
    print(f"Valid mask: {valid_mask.shape}")

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if features.shape != valid_mask.shape:
        raise ValueError(
            "Feature matrix and validity mask have different shapes."
        )

    expected_features = (
        feature_map["n_tracks"] * feature_map["n_bins"]
    )

    if features.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, "
            f"found {features.shape[1]}."
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized, statistics = normalize_features(
        features,
        valid_mask,
        feature_map,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_data = {
        "X": normalized,
        "valid_mask": valid_mask,
        "feature_map": json.dumps(feature_map),
        "normalization_stats": json.dumps(statistics),
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        **output_data,
    )

    print("\nNormalization complete.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()