# `prime-process`

The `prime-process` CLI handles data quality control, sequence deduplication, dataset splitting, and homology verification between partitions.

---

## Command Syntax

```bash
prime-process [OPTIONS]
```

---

## Options

| Option | Shorthand | Type | Required | Description |
|---|---|---|---|---|
| `--file` | `-f` | Path | **Yes** | Path to the raw input FASTA file to clean. |
| `--output` | `-o` | Path | **Yes** | Directory path where output files will be written. |
| `--name` | `-n` | String | **Yes** | Dataset/task prefix name (e.g. `Antimicrobial`, `Toxicity`). |
| `--help` | | | No | Show help message and exit. |

---

## Algorithm Details

1. **CD-HIT Clustering**:
   Redundant sequences sharing sequence identity higher than the configured threshold (default `0.7` in `Params.yaml`) are clustered, retaining only representative sequences.
2. **Stratified Splitting**:
   Sequences and labels are partitioned into stratified training and testing sets (default test size: `10%`).
3. **BLAST Homology Verification**:
   BLAST (`blastp-short`) is performed between the training and test sets to quantify cross-partition similarity and query coverage.
4. **BLAST Scatter Plot**:
   Generates a high-resolution scatter plot (`<output>/<name>_blast_hits.png`) correlating sequence identity with query coverage, with marker sizes reflecting bit scores and colors reflecting E-values.

---

## Example

```bash
prime-process \
  --file /data/raw/toxicity.fasta \
  --output results/nonr \
  --name Toxicity
```
