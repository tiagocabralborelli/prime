# PRIME

**PRIME** is a toolkit for peptide property prediction and uncertainty estimation powered by ESM (Evolutionary Scale Modeling) protein language model representations.

---

## Installation

### 1. Create and Activate Environment

Create the Conda environment using the provided environment specification:

```bash
conda env create -f tl_env.yml
conda activate tl_env
```

### 2. Install Package

Install PRIME in editable mode:

```bash
cd prime
pip install -e .
```

After installation, the three core CLI commands will be available in your terminal:
- `prime-process`
- `prime-uncert`
- `prime-predict`

---

## CLI Commands & Usage

### 1. Data Preprocessing (`prime-process`)

Cleans the raw FASTA sequence file, performs redundancy reduction using CD-HIT, splits data into non-redundant train/test partitions, and evaluates sequence homology between splits with BLAST.

```bash
prime-process \
  --file /path/to/data/amp.merged.fasta \
  --output /path/to/results/nonr \
  --name Antimicrobial
```

#### Options:
- `-f, --file FILE`: Path to the input FASTA file containing sequences and labels *(required)*.
- `-o, --output DIRECTORY`: Directory where processed partitions will be saved *(required)*.
- `-n, --name TEXT`: Prefix name for the dataset/task *(required)*.

#### Outputs:
- `<output>/<name>-train.fasta`: Training partition with non-redundant sequences.
- `<output>/<name>-test.fasta`: Test partition.
- `<output>/<name>_blast_hits.png`: BLAST identity vs. query coverage plot between train and test splits.

---

### 2. Uncertainty Estimation & Training (`prime-uncert`)

Extracts ESM embeddings (with on-disk caching), trains PyTorch classification heads (MLP / Logistic Regression), handles class imbalance with SMOTE (optional), and evaluates performance and uncertainty metrics across random seeds.

```bash
prime-uncert \
  --train /path/to/results/nonr/Antimicrobial-train.fasta \
  --test /path/to/results/nonr/Antimicrobial-test.fasta \
  --name Antimicrobial \
  --base-model /data/tiago/prime-validation/checkpoint-500000 \
  --seed 42 \
  --smote \
  --device cuda \
  --model-dir /path/to/saved_models \
  --embedding-cache-dir /path/to/embedding_cache \
  --output-dir /path/to/results/uncertainty
```

#### Options:
- `--train TEXT`: Training FASTA file with sequence labels in headers *(required)*.
- `--test TEXT`: Test FASTA file with sequence labels in headers *(required)*.
- `--name TEXT`: Task / target property name *(required)*.
- `--base-model TEXT`: Path to the base ESM model or checkpoint directory *(required)*.
- `--seed INTEGER`: Random seed for reproducibility *(default: `42`)*.
- `--smote`: Enable SMOTE oversampling for imbalanced training sets *(flag, default: disabled)*.
- `--device TEXT`: Execution device (`cuda` or `cpu`, default: auto-detected).
- `--model-dir TEXT`: Directory to save the deployable `.pkl` model artifact *(default: `saved_models`)*.
- `--embedding-cache-dir TEXT`: Directory to cache computed ESM embeddings *(default: `embedding_cache`)*.
- `--output-dir TEXT`: Directory for evaluation metrics, CSV summaries, and ROC/PR plots *(default: `results`)*.

#### Outputs:
- `<model-dir>/<name>_seed_<seed>_model.pkl`: Serialized deployable model artifact.
- `<output-dir>/seed_run_summary.csv`: Summary metrics across runs and seeds.
- `<output-dir>/bootstrap_metrics_by_seed.csv`: Bootstrapped 95% confidence intervals.
- `<output-dir>/<name>/seed_<seed>/`: Confusion matrix, ROC, and PR curves (`.png` and `.svg`).

---

### 3. Multi-Model Inference (`prime-predict`)

Runs inference on target peptide sequences using an ensemble of models (e.g. 15 seeds per feature). Embeddings are extracted once in-memory via ESM, and the final output contains the ensemble mean probability for each feature.

```bash
prime-predict \
  --input /path/to/peptides.fasta \
  --model_dir /path/to/saved_models \
  --output predictions.csv \
  --checkpoint /data/tiago/prime-validation/checkpoint-500000 \
  --device cuda \
  --include_pred \
  --include_std
```

#### Options:
- `-i, --input FILE`: Path to input FASTA file containing peptide sequences *(required)*.
- `--model_dir DIRECTORY`: Path to the directory containing `.pkl` model files *(required)*.
- `-o, --output FILE`: Output CSV file path *(default: `predictions.csv`)*.
- `-d, --device [cpu|cuda|gpu]`: Device used for embedding extraction and model evaluation *(default: `cpu`)*.
- `-c, --checkpoint TEXT`: Path to the base ESM model checkpoint *(default: `/data/tiago/prime-validation/checkpoint-500000`)*.
- `--include_pred`: Flag to include binary classification (`0` or `1`) based on the average threshold.
- `--include_std`: Flag to include epistemic standard deviation (uncertainty across model seeds).

#### Output CSV Format:
The output CSV organizes results per peptide:

| sequence_id | sequence | Allergen_prob_mean | Allergen_prob_std | Allergen_pred | Antimicrobial_prob_mean | ... |
|---|---|---|---|---|---|---|
| seq_001 | ACDEFGHIKLMNPQRSTVWY | 0.00312 | 0.00145 | 0 | 0.98540 | ... |

---

## Multi-Seed Batch Script Example

To run multi-seed uncertainty estimation across all target properties using SLURM or Bash:

```bash
SEEDS=(42 43 44 45 46 999 30 200 2 22 13 67 56 70 234)

for seed in "${SEEDS[@]}"; do
  prime-uncert \
    --train results/nonr/Antimicrobial-train.fasta \
    --test results/nonr/Antimicrobial-test.fasta \
    --name Antimicrobial \
    --base-model /data/tiago/prime-validation/checkpoint-500000 \
    --seed "${seed}" \
    --smote \
    --device cuda \
    --model-dir results/saved_models \
    --embedding-cache-dir results/embedding_cache \
    --output-dir results/uncertainty
done
```