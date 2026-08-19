from Bio import SeqIO

bad = 0
total = 0

for record in SeqIO.parse("k562_sequences_1kb.fa", "fasta"):
    seq = str(record.seq)
    total += 1
    frac_n = seq.count("N") / len(seq)
    if frac_n > 0.10:
        bad += 1

print("Total sequences:", total)
print("Sequences with >10% N:", bad)
