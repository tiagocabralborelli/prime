# Quickstart Guide

This tutorial guides you through the end-to-end PRIME workflow: preprocessing raw data, training multi-seed models, and predicting properties for new peptide sequences.

---

## Step 1: Preprocess Sequences (`prime-process`)

Prepare raw sequences by clustering with CD-HIT (to reduce sequence redundancy) and partitioning into train and test sets:

```bash
prime-process \
  --file data/antimicrobial.merged.fasta \
  --output results/nonr \
  --name Antimicrobial
```

This creates:
- `results/nonr/Antimicrobial-train.fasta`
- `results/nonr/Antimicrobial-test.fasta`
- `results/nonr/Antimicrobial_blast_hits.png`

---

## Step 2: Train with Uncertainty (`prime-uncert`)

Train the downstream classification head using ESM embeddings across multiple random seeds:

```bash
prime-uncert \
  --train results/nonr/Antimicrobial-train.fasta \
  --test results/nonr/Antimicrobial-test.fasta \
  --name Antimicrobial \
  --base-model /data/tiago/prime-validation/checkpoint-500000 \
  --seed 42 \
  --smote \
  --device cuda \
  --model-dir results/saved_models \
  --embedding-cache-dir results/embedding_cache \
  --output-dir results/uncertainty
```

Repeat this step for your chosen seeds (e.g., `42, 43, 44, 45, 46, ...`) to build an ensemble.

---

## Step 3: Run Inference (`prime-predict`)

Predict properties for novel peptide sequences using the trained models:

```bash
prime-predict \
  --input peptides_to_test.fasta \
  --model_dir results/saved_models \
  --checkpoint /data/tiago/prime-validation/checkpoint-500000 \
  --device cuda \
  --output final_predictions.csv \
  --include_pred \
  --include_std
```

The resulting `final_predictions.csv` contains the mean predicted probabilities, uncertainty standard deviations, and consensus binary predictions for every target feature.
