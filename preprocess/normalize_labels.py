import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Clip and min-max normalize ATAC accessibility labels."
    )

    parser.add_argument(
        "--input",
        default="data/processed/labels/k562_labels_raw.npy",
        help="Path to raw ATAC labels",
    )

    parser.add_argument(
        "--output",
        default="data/processed/labels/k562_labels_norm.npy",
        help="Path for normalized labels",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    labels = np.load(input_path).astype(np.float32)

    # Clip extreme outliers at the 99th percentile
    p99 = np.percentile(labels, 99)
    labels = np.clip(labels, 0, p99)

    # Min-max normalize to [0, 1]
    denominator = labels.max() - labels.min() + 1e-8
    labels_norm = (labels - labels.min()) / denominator

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels_norm.astype(np.float32))

    print("Label normalization complete")
    print("Input:", input_path)
    print("Output:", output_path)
    print("Shape:", labels_norm.shape)
    print("Min:", labels_norm.min())
    print("Max:", labels_norm.max())
    print("Mean:", labels_norm.mean())
    print("Std:", labels_norm.std())


if __name__ == "__main__":
    main()