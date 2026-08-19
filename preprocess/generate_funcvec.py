import pyBigWig
import numpy as np
import json

# -----------------------------
# INPUT FILES
# -----------------------------

bed_file = "k562_atac_peaks_1kb_windows.bed"

tracks = {
    # "ATAC": "ENCFF252GZO.bigWig",
    "H3K27ac": "H3K27ac.bigWig",
    "H3K4me3": "H3K4me3.bigWig",
    "CTCF": "CTCF.bigWig"
}

# -----------------------------
# PARAMETERS
# -----------------------------

window_size = 1000
n_bins = 16
bin_size = window_size // n_bins
n_tracks = len(tracks)

# -----------------------------
# LOAD REGIONS
# -----------------------------

regions = []
with open(bed_file) as f:
    for line in f:
        chrom, start, end = line.strip().split()[:3]
        regions.append((chrom, int(start), int(end)))

N = len(regions)

print("Total regions:", N)

# -----------------------------
# OPEN BIGWIG FILES
# -----------------------------

bw_files = {name: pyBigWig.open(path) for name, path in tracks.items()}

# -----------------------------
# INITIALIZE OUTPUT ARRAYS
# -----------------------------

feature_dim = n_tracks * n_bins

func_vectors = np.zeros((N, feature_dim))
valid_mask = np.zeros((N, feature_dim))

# -----------------------------
# PROCESS REGIONS
# -----------------------------

for i, (chrom, start, end) in enumerate(regions):

    for t, (track_name, bw) in enumerate(bw_files.items()):

        for b in range(n_bins):

            bin_start = start + b * bin_size
            bin_end = bin_start + bin_size

            feature_idx = t * n_bins + b

            try:

                vals = bw.values(chrom, bin_start, bin_end, numpy=True)

                if vals is None:
                    continue

                if np.all(np.isnan(vals)):
                    continue

                mean_signal = np.nanmean(vals)

                func_vectors[i, feature_idx] = mean_signal
                valid_mask[i, feature_idx] = 1

            except RuntimeError:
                continue

    if i % 10000 == 0:
        print("Processed", i, "regions")

# -----------------------------
# CLOSE BIGWIG FILES
# -----------------------------

for bw in bw_files.values():
    bw.close()

# -----------------------------
# SAVE OUTPUT
# -----------------------------

np.save("func_vectors_without_atac.npy", func_vectors)
np.save("valid_mask_without_atac.npy", valid_mask)

feature_map = {
    "bigwigs": list(tracks.keys()),
    "n_bins": n_bins,
    "window_size": window_size,
    "n_features": feature_dim
}

with open("feature_map.json", "w") as f:
    json.dump(feature_map, f, indent=2)

print("Functional vectors shape:", func_vectors.shape)
print("Done.")