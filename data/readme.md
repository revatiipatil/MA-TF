# MA-TF Data

This directory contains the raw and processed data required to reproduce
the MA-TF experiments.

MA-TF predicts continuous K562 chromatin accessibility using:

- DNA sequence
- H3K27ac
- H3K4me3
- CTCF

ATAC-seq is used **only as the prediction target** and must never be
included in the functional input features.

---

## 1. Data Sources

The model uses the human GRCh38/hg38 reference genome and K562
functional-genomics data.

### Reference genome

Download the UCSC hg38 reference genome:

- :contentReference[oaicite:2]{index=2}

The recommended file is:

```text
hg38.fa.gz
```

After downloading:

```bash
gunzip hg38.fa.gz
```

Place the resulting file at:

```text
data/raw/genome/hg38.fa
```

UCSC identifies hg38 as the GRCh38 human reference assembly. :contentReference[oaicite:3]{index=3}

---

## 2. ENCODE / Functional Genomic Data

The following K562 signal tracks are required.

| Track | Role | Expected filename |
|---|---|---|
| ATAC-seq | Prediction target | `ENCFF252GZO.bigWig` |
| H3K27ac | Model input | `H3K27ac.bigWig` |
| H3K4me3 | Model input | `H3K4me3.bigWig` |
| CTCF | Model input | `CTCF.bigWig` |

### ATAC-seq

The ATAC-seq target used in this project is ENCODE file:

```text
ENCFF252GZO
```

Search/download it from the official ENCODE portal:

- :contentReference[oaicite:4]{index=4}

Save the downloaded bigWig as:

```text
data/raw/encode/atac/ENCFF252GZO.bigWig
```

ATAC is used exclusively to generate the continuous accessibility
labels.

---

### H3K27ac

The K562 H3K27ac signal track is available from the UCSC hg38 ENCODE
regulation tracks:

- :contentReference[oaicite:5]{index=5}

Save it as:

```text
data/raw/encode/H3K27ac/H3K27ac.bigWig
```

---

### H3K4me3

The K562 H3K4me3 signal track is available from the UCSC hg38 ENCODE
regulation tracks:

- :contentReference[oaicite:6]{index=6}

Save it as:

```text
data/raw/encode/H3K4me3/H3K4me3.bigWig
```

The UCSC directories identify these as K562 H3K27ac and H3K4me3 signal
tracks. :contentReference[oaicite:7]{index=7}

---

### CTCF

The CTCF track should be a K562 CTCF ChIP-seq signal bigWig aligned to
GRCh38/hg38.

Official ENCODE K562 CTCF data can be located through:

- :contentReference[oaicite:8]{index=8}

For the repository's existing preprocessing, save the selected signal
track as:

```text
data/raw/encode/CTCF/CTCF.bigWig
```

**Important:** use a GRCh38/hg38 K562 CTCF signal track. Do not substitute
an hg19/hg18 track.

---

# 3. Required Directory Structure

After downloading the raw data, the repository should contain:

```text
data/
├── README.md
│
├── raw/
│   ├── genome/
│   │   └── hg38.fa
│   │
│   └── encode/
│       ├── atac/
│       │   └── ENCFF252GZO.bigWig
│       │
│       ├── H3K27ac/
│       │   └── H3K27ac.bigWig
│       │
│       ├── H3K4me3/
│       │   └── H3K4me3.bigWig
│       │
│       └── CTCF/
│           └── CTCF.bigWig
│
└── processed/
    ├── regions/
    ├── sequences/
    ├── labels/
    └── funcvecs/
```

The raw files are intentionally not stored in Git because the genome and
bigWig files are hundreds of MB to several GB.

---

# 4. Preprocessing

Run all commands from the repository root:

```bash
cd MA-TF
```

Install dependencies first:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Create genomic windows

Start with the K562 ATAC peak regions and generate the 1 kb genomic
windows used by the model.

Run:

```bash
python preprocess/make_windows.py
```

The resulting BED file should be:

```text
data/processed/regions/k562_atac_peaks_1kb_windows.bed
```

The final dataset contains approximately:

```text
123,697
```

1 kb genomic windows.

---

## Step 2 — Filter to common chromosomes

Filter the regions to chromosomes supported consistently by the
reference genome and signal tracks:

```bash
python preprocess/filter_common_chroms.py
```

The resulting regions should be stored under:

```text
data/processed/regions/
```

The project uses the canonical human chromosomes:

```text
chr1–chr22
chrX
```

---

## Step 3 — Extract DNA sequences

Extract the 1 kb DNA sequence corresponding to every genomic window:

```bash
python preprocess/extract_sequences.py
```

The resulting FASTA should be:

```text
data/processed/sequences/k562_sequences_1kb.clean.fa
```

Every sequence must have length:

```text
1000 bp
```

The sequence order must correspond exactly to the order of the BED
windows.

---

## Step 4 — Generate ATAC accessibility labels

Extract the mean ATAC signal for every 1 kb window:

```bash
python preprocess/extract_labels.py
```

Input:

```text
data/raw/encode/atac/ENCFF252GZO.bigWig
```

Output:

```text
data/processed/labels/k562_labels_raw.npy
```

The expected label array is:

```text
(123697,)
```

---

## Step 5 — Normalize labels

Normalize the accessibility targets:

```bash
python preprocess/normalize_labels.py
```

Output:

```text
data/processed/labels/k562_labels_norm.npy
```

The normalized labels are used as the regression target during model
training.

---

# 5. Generate Functional Input Features

Generate functional features from:

```text
H3K27ac
H3K4me3
CTCF
```

Run:

```bash
python preprocess/generate_funcvec.py
```

The functional feature matrix must **not contain ATAC**.

The expected output is:

```text
data/processed/funcvecs/func_vectors_without_atac.npy
```

with shape:

```text
(123697, 48)
```

The 48 features are:

```text
3 tracks × 16 bins = 48 features
```

The three tracks are:

```text
H3K27ac
H3K4me3
CTCF
```

The corresponding validity mask is:

```text
data/processed/funcvecs/valid_mask_without_atac.npy
```

with shape:

```text
(123697, 48)
```

---

# 6. ATAC Must Remain Target-Only

This is a critical requirement of the MA-TF experiment.

### Model inputs

```text
DNA sequence
H3K27ac
H3K4me3
CTCF
```

### Prediction target

```text
ATAC-seq accessibility
```

Therefore, do **not** use the ATAC bigWig when constructing the
functional feature matrix.

Use:

```text
func_vectors_without_atac.npy
```

Do not use older matrices such as:

```text
func_vectors.npy
valid_mask.npy
```

if they contain ATAC-derived features.

This prevents target leakage.

---

# 7. Verify the Dataset

Before training, verify that all modalities contain the same number of
genomic windows.

Expected:

```text
BED windows:
123697

DNA sequences:
123697

ATAC labels:
123697

Functional vectors:
123697 × 48

Functional mask:
123697 × 48
```

Run:

```bash
python preprocess/verify_data.py
```

If available for the current dataset, alignment can additionally be
checked with:

```bash
python preprocess/align_k562_datasets.py
```

Do not train if the modalities are not aligned.

---

# 8. Optional Functional Normalization

To normalize functional features:

```bash
python preprocess/normalize_func_vectors.py
```

When creating a new experiment, normalization statistics should be
computed using the training set only. Test-set information must not be
used to calculate normalization statistics.

---

# 9. Train MA-TF

After preprocessing, the model-ready data should be available under:

```text
data/processed/
```

Run the multimodal training script:

```bash
python train/train_multimodal.py
```

The training pipeline loads:

```text
DNA sequence
+
functional features
+
ATAC accessibility target
+
train/validation/test splits
```

The model architecture is defined in:

```text
train/model_multimodal.py
```

Evaluation is performed using:

```bash
python train/evaluate_multimodal.py
```

---

# 10. Data Alignment

Every row represents one genomic window.

The following must always remain aligned:

```text
BED row i
    ↓
DNA sequence i
    ↓
functional vector i
    ↓
ATAC target i
```

Do not independently shuffle or reorder individual modalities.

Train/validation/test splits should be generated from the common
sample indices and reused across all modalities.

---

# 11. Large Files and Git

The following files are intentionally excluded from the Git repository:

```text
*.bigWig
*.bw
*.fa
*.fasta
*.gz
```

This includes:

- hg38 reference genome
- ATAC-seq bigWig
- H3K27ac bigWig
- H3K4me3 bigWig
- CTCF bigWig

These files must be downloaded separately using the sources above.

The processed NumPy files may also be distributed separately if their
size exceeds normal Git repository limits.

---

# 12. Quick Reproduction Checklist

A new user should be able to reproduce the dataset using the following
workflow:

```bash
# 1. Clone repository
git clone <repository-url>
cd MA-TF

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download raw hg38 and ENCODE data
#    See the links at the top of this README.

# 4. Place raw data under data/raw/

# 5. Generate genomic windows
python preprocess/make_windows.py

# 6. Filter chromosomes
python preprocess/filter_common_chroms.py

# 7. Extract DNA sequences
python preprocess/extract_sequences.py

# 8. Extract ATAC targets
python preprocess/extract_labels.py

# 9. Normalize ATAC targets
python preprocess/normalize_labels.py

# 10. Generate functional inputs
python preprocess/generate_funcvec.py

# 11. Verify alignment
python preprocess/verify_data.py

# 12. Train MA-TF
python train/train_multimodal.py

# 13. Evaluate
python train/evaluate_multimodal.py
```

---

## Summary

MA-TF uses:

```text
DNA sequence ───────────────┐
                            │
H3K27ac ────────────────────┤
                            ├──► MA-TF ──► ATAC accessibility
H3K4me3 ────────────────────┤
                            │
CTCF ───────────────────────┘
```

ATAC-seq is the prediction target and is never used as an input
modality.