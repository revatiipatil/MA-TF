from Bio import SeqIO

input_fa = "k562_sequences_1kb.fa"
output_fa = "k562_sequences_1kb.clean.fa"

kept = 0
removed = 0

with open(output_fa, "w") as fout:
    for record in SeqIO.parse(input_fa, "fasta"):
        seq = str(record.seq)
        frac_n = seq.count("N") / len(seq)
        if frac_n <= 0.10:
            SeqIO.write(record, fout, "fasta")
            kept += 1
        else:
            removed += 1

print("Kept:", kept)
print("Removed:", removed)
print("Wrote:", output_fa)
