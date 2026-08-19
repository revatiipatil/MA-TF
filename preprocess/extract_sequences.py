from pyfaidx import Fasta

fasta = Fasta("hg38.fa")
input_bed = "k562_atac_peaks_1kb_windows.bed"
output_fasta = "k562_sequences_1kb.fa"

bad = 0
total = 0

with open(input_bed) as fin, open(output_fasta, "w") as fout:
    for i, line in enumerate(fin):
        chrom, start, end = line.strip().split("\t")
        start, end = int(start), int(end)
        total += 1

        try:
            seq = fasta[chrom][start:end].seq.upper()
        except KeyError:
            seq = "N" * (end - start)
            bad += 1

        fout.write(f">peak_{i}\n{seq}\n")

print("Total sequences:", total)
print("Missing chroms:", bad)
print("Wrote:", output_fasta)
