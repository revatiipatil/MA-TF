import numpy as np

labels = np.load("k562_labels_raw.npy")

# Clip extreme outliers (top 1%)
p99 = np.percentile(labels, 99)
labels = np.clip(labels, 0, p99)

# Min-max normalize
labels_norm = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)

np.save("k562_labels_norm.npy", labels_norm)

print("After normalization:")
print("  min:", labels_norm.min())
print("  max:", labels_norm.max())
print("  mean:", labels_norm.mean())
print("  std:", labels_norm.std())
print("Wrote: k562_labels_norm.npy")
