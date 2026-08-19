#!/usr/bin/env python3
"""
align_k562_dataset.py
---------------------

Create one fully aligned K562 multimodal dataset using the FASTA IDs
as the authoritative sample IDs.

Current source files:
    preprocess/k562_atac_peaks_1kb_windows.bed
    preprocess/k562_sequences_1kb.clean.fa
    preprocess/k562_labels_raw.npy
    preprocess/k562_labels_norm.npy
    preprocess/func_vectors_without_atac.npy
    preprocess/valid_mask_without_atac.npy

The FASTA contains 123,691 sequences while the other modalities contain
123,697 samples. Six FASTA IDs are missing.

This script:
    1. Reads BED rows and assigns peak IDs based on row order.
    2. Reads FASTA IDs and sequences.
    3. Finds the FASTA IDs that are missing.
    4. Keeps only samples that have a FASTA sequence.
    5. Filters labels, functional vectors, masks, and BED metadata
       using exactly the same indices.
    6. One-hot encodes the aligned DNA sequences.
    7. Creates deterministic train/validation/test splits.
    8. Performs strict alignment checks.
    9. Saves a clean aligned dataset.

IMPORTANT:
    We do NOT deduplicate genomic coordinates.
    peak_48863 and peak_48864 have the same coordinates but remain
    separate samples because they are separate dataset records.
"""

from pathlib import Path
import json

import numpy as np


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BED_PATH = BASE_DIR / "k562_atac_peaks_1kb_windows.bed"
FASTA_PATH = BASE_DIR / "k562_sequences_1kb.clean.fa"

RAW_LABELS_PATH = BASE_DIR / "k562_labels_raw.npy"
NORM_LABELS_PATH = BASE_DIR / "k562_labels_norm.npy"

FUNC_PATH = BASE_DIR / "func_vectors_without_atac.npy"
MASK_PATH = BASE_DIR / "valid_mask_without_atac.npy"


# Output directory
OUTPUT_DIR = BASE_DIR / "aligned_k562"

# Output files
OUTPUT_SEQUENCE = OUTPUT_DIR / "encoded_sequences_k562_1kb.npz"
OUTPUT_RAW_LABELS = OUTPUT_DIR / "labels_raw.npy"
OUTPUT_NORM_LABELS = OUTPUT_DIR / "labels_norm.npy"
OUTPUT_FUNC = OUTPUT_DIR / "func_vectors.npy"
OUTPUT_MASK = OUTPUT_DIR / "valid_mask.npy"
OUTPUT_BED = OUTPUT_DIR / "aligned_windows.bed"
OUTPUT_METADATA = OUTPUT_DIR / "metadata.npz"
OUTPUT_SPLITS = OUTPUT_DIR / "splits.npz"
OUTPUT_FEATURE_MAP = OUTPUT_DIR / "feature_map.json"
OUTPUT_REPORT = OUTPUT_DIR / "alignment_report.json"


# ============================================================
# FASTA reader
# ============================================================

def read_fasta(path):
    """
    Read FASTA while preserving sequence IDs.

    Returns:
        ids: list[str]
        sequences: list[str]
    """

    ids = []
    sequences = []

    current_id = None
    current_sequence = []

    with open(path, "r") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                # Save previous record
                if current_id is not None:
                    ids.append(current_id)
                    sequences.append("".join(current_sequence).upper())

                # FASTA header
                current_id = line[1:].split()[0]
                current_sequence = []

            else:
                current_sequence.append(line)

        # Save final record
        if current_id is not None:
            ids.append(current_id)
            sequences.append("".join(current_sequence).upper())

    return ids, sequences


# ============================================================
# BED reader
# ============================================================

def read_bed(path):
    """
    Read BED file.

    Expected:
        chr    start    end

    Peak IDs are generated from row order:
        row 0 -> peak_0
        row 1 -> peak_1
        ...
    """

    chromosomes = []
    starts = []
    ends = []

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            chromosomes.append(parts[0])
            starts.append(int(parts[1]))
            ends.append(int(parts[2]))

    n = len(chromosomes)

    ids = [f"peak_{i}" for i in range(n)]

    return (
        np.array(ids, dtype=object),
        np.array(chromosomes, dtype=object),
        np.array(starts, dtype=np.int64),
        np.array(ends, dtype=np.int64),
    )


# ============================================================
# One-hot encoding
# ============================================================

def one_hot_encode_sequence(sequence, target_length=1000):
    """
    Convert DNA sequence to shape:

        (1000, 4)

    Encoding:

        A -> [1,0,0,0]
        C -> [0,1,0,0]
        G -> [0,0,1,0]
        T -> [0,0,0,1]

    N and unknown bases remain all zeros.
    """

    sequence = sequence.upper()

    if len(sequence) != target_length:
        raise ValueError(
            f"Sequence length is {len(sequence)}, "
            f"expected {target_length}"
        )

    encoded = np.zeros(
        (target_length, 4),
        dtype=np.uint8
    )

    mapping = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
    }

    for i, base in enumerate(sequence):

        idx = mapping.get(base)

        if idx is not None:
            encoded[i, idx] = 1

    return encoded


# ============================================================
# Create splits
# ============================================================

def create_splits(n_samples, seed=42):
    """
    Create deterministic 90/5/5 train/validation/test split.
    """

    rng = np.random.RandomState(seed)

    indices = np.arange(n_samples)

    rng.shuffle(indices)

    n_train = int(0.90 * n_samples)
    n_val = int(0.05 * n_samples)

    train_idx = indices[:n_train]

    val_idx = indices[
        n_train:n_train + n_val
    ]

    test_idx = indices[
        n_train + n_val:
    ]

    return train_idx, val_idx, test_idx


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("K562 MULTIMODAL DATASET ALIGNMENT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Check source files
    # --------------------------------------------------------

    required_files = [
        BED_PATH,
        FASTA_PATH,
        RAW_LABELS_PATH,
        NORM_LABELS_PATH,
        FUNC_PATH,
        MASK_PATH,
    ]

    print("\nChecking input files...")

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        print(f"  OK: {path.name}")

    # --------------------------------------------------------
    # 2. Load BED
    # --------------------------------------------------------

    print("\nLoading BED...")

    bed_ids, chromosomes, starts, ends = read_bed(BED_PATH)

    n_bed = len(bed_ids)

    print(f"  BED samples: {n_bed}")

    # --------------------------------------------------------
    # 3. Load FASTA
    # --------------------------------------------------------

    print("\nLoading FASTA...")

    fasta_ids, sequences = read_fasta(FASTA_PATH)

    n_fasta = len(fasta_ids)

    print(f"  FASTA sequences: {n_fasta}")

    # --------------------------------------------------------
    # 4. Validate FASTA IDs
    # --------------------------------------------------------

    fasta_id_to_index = {}

    duplicate_fasta_ids = []

    for i, peak_id in enumerate(fasta_ids):

        if peak_id in fasta_id_to_index:

            duplicate_fasta_ids.append(peak_id)

        else:

            fasta_id_to_index[peak_id] = i

    if duplicate_fasta_ids:

        raise ValueError(
            "Duplicate FASTA IDs found:\n"
            + "\n".join(duplicate_fasta_ids[:20])
        )

    # --------------------------------------------------------
    # 5. Check expected IDs
    # --------------------------------------------------------

    bed_id_set = set(bed_ids)
    fasta_id_set = set(fasta_ids)

    missing_ids = sorted(
        bed_id_set - fasta_id_set,
        key=lambda x: int(x.split("_")[1])
    )

    extra_ids = sorted(
        fasta_id_set - bed_id_set,
        key=lambda x: int(x.split("_")[1])
        if x.startswith("peak_")
        and x.split("_")[1].isdigit()
        else 0
    )

    print("\nID alignment:")
    print(f"  BED IDs:       {len(bed_id_set)}")
    print(f"  FASTA IDs:     {len(fasta_id_set)}")
    print(f"  Missing FASTA: {len(missing_ids)}")
    print(f"  Extra FASTA:   {len(extra_ids)}")

    if missing_ids:

        print("\nMissing FASTA IDs:")

        for peak_id in missing_ids:
            print(f"  {peak_id}")

    if extra_ids:

        print("\nWARNING: FASTA contains IDs not present in BED:")

        for peak_id in extra_ids[:20]:
            print(f"  {peak_id}")

    # --------------------------------------------------------
    # 6. Build aligned indices
    # --------------------------------------------------------

    print("\nBuilding aligned sample indices...")

    aligned_bed_indices = []
    aligned_fasta_indices = []
    aligned_ids = []

    for bed_index, peak_id in enumerate(bed_ids):

        if peak_id not in fasta_id_to_index:
            continue

        fasta_index = fasta_id_to_index[peak_id]

        aligned_bed_indices.append(bed_index)
        aligned_fasta_indices.append(fasta_index)
        aligned_ids.append(peak_id)

    aligned_bed_indices = np.array(
        aligned_bed_indices,
        dtype=np.int64
    )

    aligned_fasta_indices = np.array(
        aligned_fasta_indices,
        dtype=np.int64
    )

    aligned_ids = np.array(
        aligned_ids,
        dtype=object
    )

    n = len(aligned_ids)

    print(f"  Final aligned samples: {n}")

    # --------------------------------------------------------
    # 7. Load labels
    # --------------------------------------------------------

    print("\nLoading labels...")

    raw_labels = np.load(RAW_LABELS_PATH)

    norm_labels = np.load(NORM_LABELS_PATH)

    print(f"  Raw labels:  {raw_labels.shape}")
    print(f"  Norm labels: {norm_labels.shape}")

    if len(raw_labels) != n_bed:

        raise ValueError(
            f"Raw labels ({len(raw_labels)}) do not match "
            f"BED ({n_bed})"
        )

    if len(norm_labels) != n_bed:

        raise ValueError(
            f"Normalized labels ({len(norm_labels)}) do not match "
            f"BED ({n_bed})"
        )

    aligned_raw_labels = raw_labels[
        aligned_bed_indices
    ].astype(np.float32)

    aligned_norm_labels = norm_labels[
        aligned_bed_indices
    ].astype(np.float32)

    # --------------------------------------------------------
    # 8. Load functional vectors
    # --------------------------------------------------------

    print("\nLoading functional vectors...")

    func_vectors = np.load(FUNC_PATH)

    valid_mask = np.load(MASK_PATH)

    print(f"  Functional vectors: {func_vectors.shape}")
    print(f"  Valid mask:         {valid_mask.shape}")

    if func_vectors.shape[0] != n_bed:

        raise ValueError(
            "Functional vector count does not match BED."
        )

    if valid_mask.shape[0] != n_bed:

        raise ValueError(
            "Valid mask count does not match BED."
        )

    aligned_func = func_vectors[
        aligned_bed_indices
    ].astype(np.float32)

    aligned_mask = valid_mask[
        aligned_bed_indices
    ].astype(np.uint8)

    # --------------------------------------------------------
    # 9. Align BED metadata
    # --------------------------------------------------------

    aligned_chromosomes = chromosomes[
        aligned_bed_indices
    ]

    aligned_starts = starts[
        aligned_bed_indices
    ]

    aligned_ends = ends[
        aligned_bed_indices
    ]

    # --------------------------------------------------------
    # 10. Align and encode sequences
    # --------------------------------------------------------

    print("\nEncoding DNA sequences...")

    aligned_sequences = [
        sequences[i]
        for i in aligned_fasta_indices
    ]

    sequence_lengths = np.array(
        [len(seq) for seq in aligned_sequences]
    )

    unique_lengths = np.unique(sequence_lengths)

    print(f"  Sequence count: {len(aligned_sequences)}")
    print(f"  Sequence lengths: {unique_lengths}")

    if not np.all(sequence_lengths == 1000):

        raise ValueError(
            "Not all aligned sequences are exactly 1000 bp."
        )

    X = np.stack(
        [
            one_hot_encode_sequence(seq, 1000)
            for seq in aligned_sequences
        ],
        axis=0
    )

    print(f"  Encoded DNA shape: {X.shape}")

    # --------------------------------------------------------
    # 11. Strict alignment validation
    # --------------------------------------------------------

    print("\nRunning alignment checks...")

    assert X.shape[0] == n
    assert aligned_raw_labels.shape[0] == n
    assert aligned_norm_labels.shape[0] == n
    assert aligned_func.shape[0] == n
    assert aligned_mask.shape[0] == n
    assert aligned_chromosomes.shape[0] == n
    assert aligned_starts.shape[0] == n
    assert aligned_ends.shape[0] == n

    # Verify IDs are exactly in the expected BED order
    expected_aligned_ids = bed_ids[
        aligned_bed_indices
    ]

    if not np.array_equal(
        aligned_ids,
        expected_aligned_ids
    ):

        raise ValueError(
            "Aligned ID ordering mismatch."
        )

    # Verify no duplicate IDs
    if len(set(aligned_ids)) != n:

        raise ValueError(
            "Duplicate IDs detected in aligned dataset."
        )

    # Verify functional dimensions
    expected_func_features = 48

    if aligned_func.shape[1] != expected_func_features:

        raise ValueError(
            f"Expected 48 functional features "
            f"(3 tracks × 16 bins), got "
            f"{aligned_func.shape[1]}"
        )

    # Verify masks
    unique_mask_values = np.unique(aligned_mask)

    if not np.all(
        np.isin(unique_mask_values, [0, 1])
    ):

        raise ValueError(
            "Valid mask must contain only 0 and 1."
        )

    print("  ✓ DNA aligned")
    print("  ✓ Raw labels aligned")
    print("  ✓ Normalized labels aligned")
    print("  ✓ Functional vectors aligned")
    print("  ✓ Valid masks aligned")
    print("  ✓ BED metadata aligned")
    print("  ✓ IDs unique")
    print("  ✓ Functional dimension = 48")
    print("  ✓ Sequence length = 1000 bp")

    # --------------------------------------------------------
    # 12. Create train/val/test splits
    # --------------------------------------------------------

    print("\nCreating train/validation/test splits...")

    train_idx, val_idx, test_idx = create_splits(
        n_samples=n,
        seed=42
    )

    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    # Check complete partition
    combined = np.concatenate(
        [train_idx, val_idx, test_idx]
    )

    if len(np.unique(combined)) != n:

        raise ValueError(
            "Train/val/test splits overlap or "
            "do not cover the complete dataset."
        )

    # --------------------------------------------------------
    # 13. Save encoded sequences
    # --------------------------------------------------------

    print("\nSaving aligned dataset...")

    np.savez_compressed(
        OUTPUT_SEQUENCE,
        X=X,
        ids=aligned_ids,
    )

    # --------------------------------------------------------
    # 14. Save labels
    # --------------------------------------------------------

    np.save(
        OUTPUT_RAW_LABELS,
        aligned_raw_labels
    )

    np.save(
        OUTPUT_NORM_LABELS,
        aligned_norm_labels
    )

    # --------------------------------------------------------
    # 15. Save functional data
    # --------------------------------------------------------

    np.save(
        OUTPUT_FUNC,
        aligned_func
    )

    np.save(
        OUTPUT_MASK,
        aligned_mask
    )

    # --------------------------------------------------------
    # 16. Save aligned BED
    # --------------------------------------------------------

    with open(OUTPUT_BED, "w") as f:

        for peak_id, chrom, start, end in zip(
            aligned_ids,
            aligned_chromosomes,
            aligned_starts,
            aligned_ends,
        ):

            f.write(
                f"{chrom}\t{start}\t{end}\t{peak_id}\n"
            )

    # --------------------------------------------------------
    # 17. Save metadata
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_METADATA,
        ids=aligned_ids,
        chromosomes=aligned_chromosomes,
        starts=aligned_starts,
        ends=aligned_ends,
    )

    # --------------------------------------------------------
    # 18. Save splits
    # --------------------------------------------------------

    np.savez(
        OUTPUT_SPLITS,
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )

    # --------------------------------------------------------
    # 19. Save feature map
    # --------------------------------------------------------

    feature_map = {
        "tracks": [
            "H3K27ac",
            "H3K4me3",
            "CTCF",
        ],
        "n_bins": 16,
        "window_size": 1000,
        "n_features": 48,
        "atac_used_as_input": False,
        "atac_used_as_target": True,
    }

    with open(
        OUTPUT_FEATURE_MAP,
        "w"
    ) as f:

        json.dump(
            feature_map,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # 20. Save alignment report
    # --------------------------------------------------------

    report = {
        "source_bed_samples": int(n_bed),
        "source_fasta_samples": int(n_fasta),
        "missing_fasta_ids": missing_ids,
        "extra_fasta_ids": extra_ids,
        "final_aligned_samples": int(n),
        "sequence_length": 1000,
        "dna_shape": list(X.shape),
        "raw_labels_shape": list(
            aligned_raw_labels.shape
        ),
        "normalized_labels_shape": list(
            aligned_norm_labels.shape
        ),
        "functional_vectors_shape": list(
            aligned_func.shape
        ),
        "valid_mask_shape": list(
            aligned_mask.shape
        ),
        "functional_tracks": [
            "H3K27ac",
            "H3K4me3",
            "CTCF",
        ],
        "functional_bins": 16,
        "atac_as_input": False,
        "atac_as_target": True,
        "split_seed": 42,
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
    }

    with open(
        OUTPUT_REPORT,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # 21. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALIGNMENT COMPLETE")
    print("=" * 70)

    print(f"""
Final dataset:

  Samples:
      {n}

  DNA:
      {X.shape}

  Raw ATAC labels:
      {aligned_raw_labels.shape}

  Normalized ATAC labels:
      {aligned_norm_labels.shape}

  Functional vectors:
      {aligned_func.shape}

  Valid mask:
      {aligned_mask.shape}

  Functional inputs:
      H3K27ac
      H3K4me3
      CTCF

  Functional features:
      3 tracks × 16 bins = 48

  ATAC:
      TARGET ONLY
      NOT USED AS INPUT

Splits:

  Train:
      {len(train_idx)}

  Validation:
      {len(val_idx)}

  Test:
      {len(test_idx)}

Output directory:
    {OUTPUT_DIR}

Missing FASTA records excluded:
    {len(missing_ids)}
""")

    print("Files created:")

    for path in [
        OUTPUT_SEQUENCE,
        OUTPUT_RAW_LABELS,
        OUTPUT_NORM_LABELS,
        OUTPUT_FUNC,
        OUTPUT_MASK,
        OUTPUT_BED,
        OUTPUT_METADATA,
        OUTPUT_SPLITS,
        OUTPUT_FEATURE_MAP,
        OUTPUT_REPORT,
    ]:

        print(f"  {path.name}")

    print("\nSUCCESS: All modalities are aligned.")


if __name__ == "__main__":
    main()