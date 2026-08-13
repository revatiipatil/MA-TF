valid_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

input_bed = "E:\\chromatin_accessibility\\data\\processed\\peaks\\full_peaks.bed"
output_bed = "E:\\chromatin_accessibility\\data\\processed\\peaks\\full_peaks.commonchroms.bed"

kept = 0
total = 0

with open(input_bed) as fin, open(output_bed, "w") as fout:
    for line in fin:
        total += 1
        if not line.strip():
            continue
        chrom = line.split("\t")[0]
        if chrom in valid_chroms:
            fout.write(line)
            kept += 1

print(f"Total peaks: {total}")
print(f"Kept peaks (standard chroms): {kept}")
print(f"Removed peaks (alt contigs): {total - kept}")
print("Wrote:", output_bed)
