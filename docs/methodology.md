# Methodology & Architecture

This document outlines the theoretical and architectural design behind PRIME.

---

## 1. Sequence Representation (ESM Embeddings)

Traditional peptide feature engineering relies on hand-crafted physicochemical properties (such as charge, hydrophobicity, or amino acid composition). 

PRIME utilizes **ESM-2 (Evolutionary Scale Modeling)**, a transformer-based masked language model trained on millions of evolutionary protein sequences:
- **Tokenization**: Sequences are mapped into residue tokens with start/stop boundary tokens.
- **Hidden Representations**: The last hidden layer captures high-level context, secondary structure tendencies, and evolutionary conservation.
- **Pooling**: An attention-masked mean pooling operation collapses residue embeddings across sequence length into a fixed-size embedding vector (e.g. 320 dimensions).

---

## 2. Redundancy Control

To prevent optimistic bias and test-set leakage, `prime-process` applies strict sequence homology partitioning:
- **CD-HIT**: Eliminates near-identical duplicates above a specified identity threshold (e.g., 70%).
- **BLAST Identity Check**: Verifies cross-partition coverage and alignment metrics to ensure test sequences test generalizability rather than memorization.

---

## 3. Epistemic Uncertainty Estimation

Predictions in drug discovery and therapeutic peptide design require confidence estimates:
- **Multi-Seed Variance**: Models trained with different random seeds capture variance introduced by initialization, cross-validation folds, and SMOTE resampling.
- **Monte Carlo Dropout**: During inference or evaluation, dropout remains active across passes to evaluate parameter posterior uncertainty.
- **Bootstrap Validation**: Non-parametric bootstrap resampling (e.g. 1,000 iterations) generates 95% confidence intervals for ROC-AUC, PR-AUC, Accuracy, F1-score, and Matthews Correlation Coefficient.

---

## 4. Multi-Model Ensemble

During inference (`prime-predict`), each peptide is evaluated across $M$ models (typically $M=15$ seeds per property):

$$\bar{P}_i = \frac{1}{M} \sum_{m=1}^{M} P_{i,m}$$

$$\sigma_i = \sqrt{\frac{1}{M-1} \sum_{m=1}^{M} (P_{i,m} - \bar{P}_i)^2}$$

Where:
- $\bar{P}_i$ is the ensemble mean probability for sequence $i$.
- $\sigma_i$ quantifies model epistemic disagreement (uncertainty).
