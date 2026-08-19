import pyBigWig
import numpy as np

bw = pyBigWig.open("ENCFF252GZO.bigWig")

input_bed = "k562_atac_peaks_1kb_windows.bed"
output_labels = "k562_labels_raw.npy"

labels = []
bad = 0
total = 0

with open(input_bed) as fin:
    for line in fin:
        chrom, start, end = line.strip().split("\t")
        start, end = int(start), int(end)
        total += 1

        try:
            values = bw.values(chrom, start, end, numpy=True)
            values = np.array(values, dtype=np.float32)
            values = values[~np.isnan(values)]

            if len(values) == 0:
                labels.append(0.0)
                bad += 1
            else:
                labels.append(values.mean())

        except RuntimeError:
            labels.append(0.0)
            bad += 1

bw.close()

labels = np.array(labels, dtype=np.float32)
np.save(output_labels, labels)

print("Total windows:", total)
print("Bad windows (no signal):", bad)
print("Label stats:")
print("  min:", labels.min())
print("  max:", labels.max())
print("  mean:", labels.mean())
print("  std:", labels.std())
print("Wrote:", output_labels)
