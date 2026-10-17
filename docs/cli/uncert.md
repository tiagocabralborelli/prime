# `prime-uncert`

The `prime-uncert` CLI performs embedding-based training of PyTorch classifiers, epistemic uncertainty estimation, bootstrap evaluations, and model serialization.

---

## Command Syntax

```bash
prime-uncert [OPTIONS]
```

---

## Options

| Option | Type | Default | Required | Description |
|---|---|---|---|---|
| `--train` | Path | — | **Yes** | Training FASTA file with sequence labels encoded in headers. |
| `--test` | Path | — | **Yes** | Test FASTA file with sequence labels encoded in headers. |
| `--name` | String | — | **Yes** | Project or target property name (e.g. `Hemolitic`). |
| `--base-model` | Path | — | **Yes** | Path to the pre-trained ESM base model or checkpoint directory. |
| `--seed` | Integer | `42` | No | Random seed for data splitting and model initialization. |
| `--smote` | Flag | `False` | No | Enables SMOTE oversampling on training embeddings. |
| `--device` | String | `cuda` / `cpu` | No | Compute device for embedding extraction and training. |
| `--model-dir` | Path | `saved_models` | No | Directory to save the deployable model artifact (`.pkl`). |
| `--embedding-cache-dir`| Path | `embedding_cache`| No | Directory to cache raw embeddings as `.npz` files. |
| `--output-dir` | Path | `results` | No | Directory where evaluation tables and plots are stored. |

---

## Outputs

1. **Deployable Model Artifact (`<model-dir>/<name>_seed_<seed>_model.pkl`)**:
   Contains the trained PyTorch MLP / Logistic Regression model, StandardScaler, optimal decision threshold, class mapping, and ESM checkpoint metadata.
2. **Metrics Summaries (`<output-dir>/`)**:
   - `seed_run_summary.csv`: Execution summary across seeds.
   - `bootstrap_metrics_by_seed.csv`: 95% bootstrap confidence intervals for ROC-AUC, PR-AUC, Accuracy, F1, and MCC.
   - `seed_predictions.csv`: Individual sequence probabilities per seed.
3. **Evaluation Plots (`<output-dir>/<name>/seed_<seed>/`)**:
   - Confusion matrix (`.png` and `.svg`)
   - ROC Curve (`.png` and `.svg`)
   - Precision-Recall Curve (`.png` and `.svg`)

---

## Example

```bash
prime-uncert \
  --train results/nonr/Hemolitic-train.fasta \
  --test results/nonr/Hemolitic-test.fasta \
  --name Hemolitic \
  --base-model /data/tiago/prime-validation/checkpoint-500000 \
  --seed 200 \
  --smote \
  --device cuda \
  --model-dir results/saved_models \
  --embedding-cache-dir results/embedding_cache \
  --output-dir results/uncertainty
```
