#!/usr/bin/env python3
"""
Generate functional genomic feature vectors for MA-TF.

Input:
    data/processed/regions/k562_atac_peaks_1kb_windows.bed

Functional tracks:
    - H3K27ac
    - H3K4me3
    - CTCF

ATAC-seq is intentionally excluded from the input features because
ATAC-seq is the prediction target.

For each 1 kb genomic window, each track is divided into 16 bins.
The mean signal in each bin is used as the feature.

Output:
    data/processed/functional/functional_features.npz

The output NPZ contains:
    X           : functional feature matrix, shape (N, 48)
    valid_mask  : signal-validity mask, shape (N, 48)
    feature_map: metadata describing tracks and bins
"""

from pathlib import Path
import json

import numpy as np
import pyBigWig


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "regions"
    / "k562_atac_peaks_1kb_windows.bed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "functional"
)

OUTPUT_FILE = OUTPUT_DIR / "functional_features.npz"


# ============================================================
# Functional tracks
# ============================================================

TRACKS = {
    "H3K27ac": PROJECT_ROOT / "data" / "raw" / "encode" / "H3K27ac" / "H3K27ac.bigWig",
    "H3K4me3": PROJECT_ROOT / "data" / "raw" / "encode" / "H3K4me3" / "H3K4me3.bigWig",
    "CTCF": PROJECT_ROOT / "data" / "raw" / "encode" / "CTCF" / "CTCF.bigWig",
}


# ============================================================
# Parameters
# ============================================================

WINDOW_SIZE = 1000
N_BINS = 16
BIN_SIZE = WINDOW_SIZE // N_BINS


# ============================================================
# Load genomic regions
# ============================================================

def load_regions(bed_file: Path):
    regions = []

    with bed_file.open() as f:
        for line in f:
            if not line.strip():
                continue

            chrom, start, end = line.strip().split()[:3]
            regions.append(
                (chrom, int(start), int(end))
            )

    return regions


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MA-TF Functional Feature Generation")
    print("=" * 60)

    print("\nBED file:")
    print(BED_FILE)

    if not BED_FILE.exists():
        raise FileNotFoundError(
            f"BED file not found: {BED_FILE}"
        )

    # --------------------------------------------------------
    # Load regions
    # --------------------------------------------------------

    regions = load_regions(BED_FILE)
    n_regions = len(regions)

    print(f"\nTotal regions: {n_regions}")

    # --------------------------------------------------------
    # Check BigWig files
    # --------------------------------------------------------

    print("\nChecking functional tracks:")

    for name, path in TRACKS.items():

        print(f"  {name}: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"BigWig file not found: {path}"
            )

    # --------------------------------------------------------
    # Open BigWigs
    # --------------------------------------------------------

    bw_files = {
        name: pyBigWig.open(str(path))
        for name, path in TRACKS.items()
    }

    feature_dim = len(TRACKS) * N_BINS

    func_vectors = np.zeros(
        (n_regions, feature_dim),
        dtype=np.float32,
    )

    valid_mask = np.zeros(
        (n_regions, feature_dim),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Extract binned signal
    # --------------------------------------------------------

    for i, (chrom, start, end) in enumerate(regions):

        for track_idx, (track_name, bw) in enumerate(
            bw_files.items()
        ):

            for bin_idx in range(N_BINS):

                bin_start = (
                    start + bin_idx * BIN_SIZE
                )

                bin_end = (
                    bin_start + BIN_SIZE
                )

                feature_idx = (
                    track_idx * N_BINS + bin_idx
                )

                try:

                    values = bw.values(
                        chrom,
                        bin_start,
                        bin_end,
                        numpy=True,
                    )

                    if values is None:
                        continue

                    values = np.asarray(
                        values,
                        dtype=np.float32,
                    )

                    if values.size == 0:
                        continue

                    if np.all(np.isnan(values)):
                        continue

                    func_vectors[
                        i,
                        feature_idx
                    ] = np.nanmean(values)

                    valid_mask[
                        i,
                        feature_idx
                    ] = 1

                except RuntimeError:
                    continue

        if i % 10000 == 0:
            print(
                f"Processed {i}/{n_regions} regions"
            )

    # --------------------------------------------------------
    # Close BigWigs
    # --------------------------------------------------------

    for bw in bw_files.values():
        bw.close()

    # --------------------------------------------------------
    # Feature metadata
    # --------------------------------------------------------

    feature_map = {
        "tracks": list(TRACKS.keys()),
        "n_tracks": len(TRACKS),
        "window_size": WINDOW_SIZE,
        "n_bins": N_BINS,
        "bin_size": BIN_SIZE,
        "n_features": feature_dim,
        "atac_included": False,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=func_vectors,
        valid_mask=valid_mask,
        feature_map=json.dumps(feature_map),
    )

    print("\n" + "=" * 60)
    print("Functional feature generation complete")
    print("=" * 60)

    print(f"Features:    {func_vectors.shape}")
    print(f"Valid mask:  {valid_mask.shape}")
    print(f"Output:      {OUTPUT_FILE}")


if __name__ == "__main__":
    main()