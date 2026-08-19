WINDOW = 1000
HALF = WINDOW // 2

input_bed = "k562_atac_peaks.commonchroms.bed"
output_bed = "k562_atac_peaks_1kb_windows.bed"

with open(input_bed) as fin, open(output_bed, "w") as fout:
    for line in fin:
        fields = line.strip().split("\t")
        chrom, start, end = fields[0], int(fields[1]), int(fields[2])
        center = (start + end) // 2
        new_start = max(0, center - HALF)
        new_end = center + HALF
        fout.write(f"{chrom}\t{new_start}\t{new_end}\n")

print("Wrote:", output_bed)
