#!/usr/bin/env python3
"""
Summarize bigWig signals into fixed-length functional feature vectors.

Inputs:
  - BED with regions (chrom, start, end[, region_id...])
  - One or more bigWig files (ATAC signal, histone marks, etc.)

Outputs:
  - func_vectors.npy:  (N, len(bigwigs)*n_bins)
  - valid_mask.npy:    (N, len(bigwigs)*n_bins)  1=covered, 0=NaN
  - feature_map.json:  names and indices of each feature
  - (optional) access_target.npy: (N, n_bins) from a chosen ATAC bigWig

Usage (your tree):
  python summarize_bigwigs.py \
      --bed data/processed/peaks/full_peaks.bed \
      --bigwig raw/encode/atac_seq/K562/atac_signal.bigWig \
      --bigwig raw/encode/chip_seq/K562/H3K4me3.bigWig \
      --bigwig raw/encode/chip_seq/K562/H3K27ac.bigWig \
      --bigwig raw/encode/chip_seq/K562/H3K27me3.bigWig \
      --n-bins 16 \
      --out-dir data/processed/funcvecs \
      --make-access-target-from 0  # 0 means the first bigWig (ATAC)
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np


def read_bed(path: str):
    """Reads BED minimally as chrom, start, end, with optional region_id."""
    cols = []
    data = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            region_id = parts[3] if len(parts) >= 4 else f"{chrom}:{start}-{end}"
            data.append((chrom, start, end, region_id))
    if not data:
        raise ValueError(f"No BED intervals parsed from: {path}")
    arr = np.array(data, dtype=object)
    return arr  # shape (N, 4): chrom, start, end, region_id


def summarize_one_bw(bw_path: str, bed_arr: np.ndarray, n_bins: int):
    """Return (N, n_bins) mean per bin using pyBigWig.stats; NaNs preserved.
       Robust to chr-prefix mismatches and out-of-range coords."""
    try:
        import pyBigWig
    except Exception as e:
        print("ERROR: pyBigWig is required. pip install pyBigWig", file=sys.stderr)
        raise e

    bw = pyBigWig.open(bw_path)
    if bw is None or not bw.isBigWig():
        raise ValueError(f"Cannot open bigWig: {bw_path}")

    chrom_sizes = bw.chroms()  # dict: chrom -> length

    def normalize_chrom(c: str):
        if c in chrom_sizes:
            return c
        if c.startswith("chr"):
            c2 = c[3:]
            if c2 in chrom_sizes:
                return c2
        else:
            c2 = "chr" + c
            if c2 in chrom_sizes:
                return c2
        # MT/chrM aliasing
        aliases = {"MT": ["chrM", "M", "chrMT"], "chrM": ["MT", "M", "chrMT"]}
        for k, vals in aliases.items():
            if c == k or c in vals:
                for v in [k] + vals:
                    if v in chrom_sizes:
                        return v
        return None

    N = bed_arr.shape[0]
    out = np.zeros((N, n_bins), dtype=np.float32)
    with_nan = np.zeros((N, n_bins), dtype=bool)

    for i, (chrom, start, end, _) in enumerate(bed_arr):
        c = normalize_chrom(str(chrom))
        if c is None:
            out[i, :] = np.nan
            with_nan[i, :] = True
            continue

        L = chrom_sizes[c]
        s = max(0, int(start))
        e = min(int(end), L)
        if e <= s:
            out[i, :] = np.nan
            with_nan[i, :] = True
            continue

        try:
            vals = bw.stats(c, s, e, nBins=n_bins, type="mean")
        except RuntimeError:
            vals = [None] * n_bins

        v = np.array([np.nan if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)
                      for x in vals], dtype=np.float32)
        out[i, :] = v
        with_nan[i, :] = np.isnan(v)

    bw.close()
    return out, with_nan



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True, help="BED file with regions")
    ap.add_argument(
        "--bigwig",
        action="append",
        required=True,
        help="Path to a bigWig (repeat for multiple). Order matters.",
    )
    ap.add_argument("--n-bins", type=int, default=16, help="Number of bins per region")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--make-access-target-from",
        type=int,
        default=None,
        help="Index of bigWig to use for access_target.npy (e.g., 0 for ATAC).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bed_arr = read_bed(args.bed)
    N = bed_arr.shape[0]
    K = args.n_bins
    B = len(args.bigwig)

    print(f"[INFO] Regions: {N}, Bins: {K}, BigWigs: {B}")

    # Summarize each bigWig to (N, K)
    per_bw = []
    per_bw_nan = []
    for idx, bw_path in enumerate(args.bigwig):
        print(f"[INFO] Summarizing {idx}: {bw_path}")
        X, Xnan = summarize_one_bw(bw_path, bed_arr, K)
        per_bw.append(X)
        per_bw_nan.append(Xnan)

    # Concatenate into (N, B*K)
    func_vectors = np.concatenate(per_bw, axis=1).astype(np.float32)
    nan_mask = np.concatenate(per_bw_nan, axis=1)  # True where NaN

    # Build validity mask (1=valid, 0=missing) and fill NaNs with 0
    valid_mask = (~nan_mask).astype(np.uint8)
    func_vectors = np.nan_to_num(func_vectors, nan=0.0)

    # Feature map
    feature_list = []
    col = 0
    for b_idx, bw_path in enumerate(args.bigwig):
        base = os.path.basename(bw_path)
        for k in range(K):
            feature_list.append(
                {"name": f"{base}:bin_{k}_mean", "index": col, "src": base}
            )
            col += 1
    feature_map = {
        "n_bins": K,
        "n_features": B * K,
        "bigwigs": args.bigwig,
        "features": feature_list,
        "bed_source": os.path.abspath(args.bed),
    }

    # Save outputs
    np.save(os.path.join(args.out_dir, "func_vectors.npy"), func_vectors)
    np.save(os.path.join(args.out_dir, "valid_mask.npy"), valid_mask)
    with open(os.path.join(args.out_dir, "feature_map.json"), "w") as f:
        json.dump(feature_map, f, indent=2)

    print(f"[OK] Saved: {os.path.join(args.out_dir, 'func_vectors.npy')}  shape={func_vectors.shape}")
    print(f"[OK] Saved: {os.path.join(args.out_dir, 'valid_mask.npy')}    shape={valid_mask.shape}")
    print(f"[OK] Saved: {os.path.join(args.out_dir, 'feature_map.json')}")

    # Optional: accessibility target from a chosen bigWig (e.g., ATAC)
    if args.make_access_target_from is not None:
        src = args.make_access_target_from
        if not (0 <= src < B):
            raise ValueError("--make-access-target-from must be in [0, B-1]")
        access = per_bw[src].astype(np.float32)
        access_mask = (~per_bw_nan[src]).astype(np.uint8)
        access = np.nan_to_num(access, nan=0.0)

        np.save(os.path.join(args.out_dir, "access_target.npy"), access)
        np.save(os.path.join(args.out_dir, "access_valid_mask.npy"), access_mask)
        print(f"[OK] Saved: access_target.npy  shape={access.shape}")
        print(f"[OK] Saved: access_valid_mask.npy  shape={access_mask.shape}")


if __name__ == "__main__":
    main()
