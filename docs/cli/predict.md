# `prime-predict`

The `prime-predict` CLI provides multi-model ensemble inference on target peptide sequences. It extracts ESM representations once in-memory and averages predictions across all available seed models for each property.

---

## Command Syntax

```bash
prime-predict [OPTIONS]
```

---

## Options

| Option | Shorthand | Type | Default | Required | Description |
|---|---|---|---|---|---|
| `--input` | `-i` | Path | — | **Yes** | Input FASTA file containing candidate peptide sequences. |
| `--model_dir` | `--model-dir` | Path | — | **Yes** | Directory containing the trained `.pkl` model artifacts. |
| `--output` | `-o` | Path | `predictions.csv` | No | Path to the destination CSV file for predictions. |
| `--device` | `-d` | Choice | `cpu` | No | Compute device: `cpu`, `cuda`, or `gpu`. |
| `--checkpoint`| `-c` | Path | `/data/.../checkpoint-500000` | No | Path to the ESM base model checkpoint directory. |
| `--include_pred`| `--include-pred` | Flag | `False` | No | Include binary classification (`0` or `1`) based on mean threshold. |
| `--include_std` | `--include-std` | Flag | `False` | No | Include standard deviation (epistemic uncertainty) across seeds. |

---

## Features Discovered Automatically

When pointing `--model_dir` to a directory containing models named like:
```text
Antimicrobial_old_celio_merged_seed_42_model.pkl
Toxicity_old_celio_merged_seed_42_model.pkl
Hemolitic_old_celio_merged_seed_42_model.pkl
```

The CLI automatically parses the prefix before the first underscore (`_`), groups the models into feature ensembles (e.g. 15 models per feature), and runs predictions across all features in a single pass.

---

## Output CSV Structure

| Column | Description |
|---|---|
| `sequence_id` | Identifier extracted from the FASTA header (`>id`). |
| `sequence` | Canonical amino acid sequence. |
| `<Feature>_prob_mean` | Ensemble mean probability for the positive class (e.g. `0.98540`). |
| `<Feature>_prob_std` | *(Optional, if `--include_std`)* Standard deviation across the 15 models. |
| `<Feature>_pred` | *(Optional, if `--include_pred`)* Binary consensus decision (`1` if prob $\ge$ threshold, else `0`). |

---

## Example

```bash
prime-predict \
  --input novel_candidates.fasta \
  --model_dir /data/saved_models \
  --checkpoint /data/checkpoint-500000 \
  --device gpu \
  --output final_results.csv \
  --include_pred \
  --include_std
```
