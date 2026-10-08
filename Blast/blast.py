#!/usr/bin/env python3
r"""
Nine-task peptide dataset similarity pipeline using NCBI BLAST+.

Leakage detection between query and target datasets.

High-similarity definition (single threshold, no coarse pre-filter):
    identity >= 80% AND query coverage >= 80% AND target coverage >= 80%

Only this criterion is used. No intermediate qcov >= 50 gate.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================================
# USER CONFIGURATION
# ============================================================================
DATASETS = {
    "Toxicity": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\toxitity-train.fasta",
        "old_test": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\toxicity-test.fasta",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\toxicity\toxic_train.fasta",
        "new_test": r"D:\Desktop\AI\CAopt-main\data-seq\toxicity\toxic_test.fasta",
    },
    "Allergen": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\allergen-train.fasta",
        "old_test": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\allergen-test.fasta",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\allergen\train_merged_allergen--2.fasta",
        "new_test": r"D:\Desktop\AI\CAopt-main\data-seq\allergen\test_merged_allergen--2.fasta",
    },
    "AMP": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\amp-train.fasta",
        "old_test": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\amp-test.fasta",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\data-shuffle\amp.train-3_shuffle.fasta",
        "new_test": r"D:\Desktop\AI\CAopt-main\data-seq\data-shuffle\amp.test-3_shuffle.fasta",
    },
    "AOX": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\anoxpepred_testset.fasta",
        "old_test": r"",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\data-shuffle\train-Antiox--4..fasta",
        "new_test": r"",
    },
    "Fouling": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\fouling-train..fasta",
        "old_test": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\fouling-test..fasta",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\fouling\fouling-train-shuffle-1.fasta",
        "new_test": r"D:\Desktop\AI\CAopt-main\data-seq\fouling\fouling-test-shuffle-1.fasta",
    },
    "Hemolytic": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\hemo-train.fasta",
        "old_test": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\hemo-test.fasta",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\Hemo\hemolysis_train.fasta",
        "new_test": r"D:\Desktop\AI\CAopt-main\data-seq\Hemo\hemolysis_test.fasta",
    },
    "QSI": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\qsi-train_test.fasta",
        "old_test": r"",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\data-shuffle\train-QS-9 shuffle.fasta",
        "new_test": r"",
    },
    "Solubility": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\sol-train.fasta",
        "old_test": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\sol-test..fasta",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\data-shuffle\solubility-train-101.fasta",
        "new_test": r"D:\Desktop\AI\CAopt-main\data-seq\data-shuffle\solubility-test-101.fasta",
    },
    "Stability": {
        "old_train": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\stability-train..fasta",
        "old_test": r"D:\Desktop\AI\CAopt-main\data-seq\6.7上传数据\6月17日\stability-test_2class..fasta",
        "new_train": r"D:\Desktop\AI\CAopt-main\data-seq\data-shuffle\stability-train-11.fasta",
        "new_test": r"D:\Desktop\AI\CAopt-main\data-seq\data-shuffle\stability-test-11.fasta",
    },
}

OUTPUT_ROOT = Path(r"D:\Desktop\Model\similarity_results-3-2")

THREADS = 8
BLASTP_EXECUTABLE = r"D:\Pycharm\blast\blast-2.17.0+\bin\blastp.exe"
MAKEBLASTDB_EXECUTABLE = r"D:\Pycharm\blast\blast-2.17.0+\bin\makeblastdb.exe"
MAX_TARGET_SEQS = 10000

COMPARISONS = {
    "old_test_vs_old_train": ("old_test", "old_train"),
    "new_test_vs_new_train": ("new_test", "new_train"),
    "new_test_vs_old_test": ("new_test", "old_test"),
}

BLAST_COLUMNS = [
    "query", "target", "pident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "qlen", "tstart", "tend", "tlen", "evalue", "bits",
]

# Single threshold used everywhere. No coarse qcov gate.
SEARCH_EVALUE = 1e-3
HIGH_IDENTITY = 80.0
HIGH_QUERY_COVERAGE = 80.0
HIGH_TARGET_COVERAGE = 80.0

ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")


@dataclass(frozen=True)
class FastaRecord:
    seq_id: str
    sequence: str


@dataclass
class FastaReadResult:
    records: List[FastaRecord]
    skipped_invalid: List[Tuple[str, List[str]]]
    skipped_empty: List[str]
    skipped_duplicate: List[str]

    @property
    def total_skipped(self) -> int:
        return len(self.skipped_invalid) + len(self.skipped_empty) + len(self.skipped_duplicate)

    @property
    def total_headers(self) -> int:
        return len(self.records) + self.total_skipped


def read_fasta(path: Path) -> FastaReadResult:
    records: List[FastaRecord] = []
    skipped_invalid: List[Tuple[str, List[str]]] = []
    skipped_empty: List[str] = []
    skipped_duplicate: List[str] = []
    current_id = None
    pieces: List[str] = []
    seen = set()

    def commit() -> None:
        nonlocal current_id, pieces
        if current_id is None:
            return
        sequence = "".join(pieces).replace(" ", "").upper()
        if not sequence:
            skipped_empty.append(current_id)
        else:
            invalid = sorted(set(sequence) - ALLOWED_AA)
            if invalid:
                skipped_invalid.append((current_id, invalid))
            elif current_id in seen:
                skipped_duplicate.append(current_id)
            else:
                seen.add(current_id)
                records.append(FastaRecord(current_id, sequence))
        current_id = None
        pieces = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                commit()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Blank FASTA header at {path}:{line_number}")
                current_id = header.split()[0]
            else:
                if current_id is None:
                    raise ValueError(f"Sequence before first header at {path}:{line_number}")
                pieces.append(line)
    commit()
    if not records:
        raise ValueError(f"No valid FASTA records found in {path}")
    return FastaReadResult(records, skipped_invalid, skipped_empty, skipped_duplicate)


def write_skipped_details(result: FastaReadResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for seq_id, chars in result.skipped_invalid:
            handle.write(f"INVALID\t{seq_id}\t{','.join(chars)}\n")
        for seq_id in result.skipped_empty:
            handle.write(f"EMPTY\t{seq_id}\n")
        for seq_id in result.skipped_duplicate:
            handle.write(f"DUPLICATE\t{seq_id}\n")


def calculate_length_statistics(activity: str, dataset_name: str, path: Path,
                                records: List[FastaRecord]) -> dict:
    lengths = np.asarray([len(r.sequence) for r in records], dtype=int)
    n = len(lengths)
    return {
        "activity": activity, "dataset": dataset_name, "fasta_path": str(path),
        "sequence_count": n, "minimum_length": int(lengths.min()),
        "q25_length": float(np.percentile(lengths, 25)),
        "median_length": float(np.median(lengths)), "mean_length": float(lengths.mean()),
        "q75_length": float(np.percentile(lengths, 75)), "maximum_length": int(lengths.max()),
        "length_lt8_n": int((lengths < 8).sum()),
        "length_lt8_percent": 100.0 * float((lengths < 8).sum()) / n,
        "length_lt15_n": int((lengths < 15).sum()),
        "length_lt15_percent": 100.0 * float((lengths < 15).sum()) / n,
        "length_lt30_n": int((lengths < 30).sum()),
        "length_lt30_percent": 100.0 * float((lengths < 30).sum()) / n,
    }


def validate_dataset_paths() -> Dict[str, Dict[str, Path | None]]:
    required = {"old_train", "old_test", "new_train", "new_test"}
    validated: Dict[str, Dict[str, Path | None]] = {}
    problems: List[str] = []
    for activity, mapping in DATASETS.items():
        validated[activity] = {}
        for key in sorted(required):
            value = mapping.get(key, "")
            if not str(value).strip():
                validated[activity][key] = None
                continue
            path = Path(value).expanduser()
            if not path.is_file():
                problems.append(f"{activity}.{key}: file not found: {path}")
            else:
                validated[activity][key] = path.resolve()
    if problems:
        raise FileNotFoundError("Please correct DATASETS:\n  - " + "\n  - ".join(problems))
    return validated


def write_fasta(records: List[FastaRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.seq_id}\n{record.sequence}\n")


def run_blast(executable: str, makeblastdb_executable: str,
              query_records: List[FastaRecord], target_records: List[FastaRecord],
              output_tsv: Path, tmp_dir: Path, threads: int,
              max_target_seqs: int) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_target = tmp_dir / "target.fasta"
    write_fasta(target_records, safe_target)

    db_prefix = tmp_dir / "target_db"
    db_command = [
        makeblastdb_executable, "-in", str(safe_target), "-dbtype", "prot",
        "-out", str(db_prefix),
    ]
    print("RUN:", " ".join(db_command), flush=True)
    subprocess.run(db_command, check=True)

    groups = {"standard": query_records}

    part_outputs: List[Path] = []
    outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend qlen sstart send slen evalue bitscore"
    for group_name, records in groups.items():
        if not records:
            continue
        query_path = tmp_dir / f"query_{group_name}.fasta"
        part_output = tmp_dir / f"blast_{group_name}.tsv"
        write_fasta(records, query_path)
        command = [
            executable, "-task", "blastp",
            "-query", str(query_path), "-db", str(db_prefix), "-out", str(part_output),
            "-outfmt", outfmt, "-evalue", str(SEARCH_EVALUE),
            "-max_target_seqs", str(max_target_seqs), "-seg", "no",
            "-num_threads", str(threads),
        ]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        part_outputs.append(part_output)

    with output_tsv.open("wb") as destination:
        for part_output in part_outputs:
            if part_output.exists():
                destination.write(part_output.read_bytes())


def load_all_hits(path: Path) -> pd.DataFrame:
    """Read raw BLAST output. No coarse qcov gate; all hits are kept."""
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=BLAST_COLUMNS)
    df = pd.read_csv(path, sep="\t", names=BLAST_COLUMNS,
                     dtype={"query": str, "target": str})
    numeric = [c for c in BLAST_COLUMNS if c not in {"query", "target"}]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["query", "target", "pident", "qlen", "tlen", "evalue", "bits"])
    df["qcov"] = 100.0 * df["alnlen"] / df["qlen"]
    df["tcov"] = 100.0 * df["alnlen"] / df["tlen"]
    return df


def choose_best_hits(raw: pd.DataFrame, query_records: List[FastaRecord],
                     target_records: List[FastaRecord]) -> pd.DataFrame:
    """
    Pick, for each query, the hit that best satisfies identity >= 80,
    qcov >= 80 and tcov >= 80. If none satisfies all three, the query is
    treated as 'no high-similarity hit' for leakage purposes.
    """
    query_table = pd.DataFrame({
        "query": [r.seq_id for r in query_records],
        "query_sequence": [r.sequence for r in query_records],
    })

    if raw.empty:
        # Build an empty frame with the columns merge and downstream code expect.
        best = pd.DataFrame(columns=[
            "query", "target", "pident", "alnlen", "mismatch", "gapopen",
            "qstart", "qend", "qlen", "tstart", "tend", "tlen",
            "evalue", "bits", "qcov", "tcov",
        ])
    else:
        candidates = raw.copy()
        candidates["qualifies"] = (
            (candidates["pident"] >= HIGH_IDENTITY)
            & (candidates["qcov"] >= HIGH_QUERY_COVERAGE)
            & (candidates["tcov"] >= HIGH_TARGET_COVERAGE)
        )
        ordered = candidates.sort_values(
            ["query", "qualifies", "pident", "qcov", "tcov", "bits", "evalue", "target"],
            ascending=[True, False, False, False, False, False, True, True],
            kind="mergesort",
        )
        best = ordered.drop_duplicates("query", keep="first")

    result = query_table.merge(best, on="query", how="left", validate="one_to_one")
    target_sequences = {r.seq_id: r.sequence for r in target_records}
    result.insert(result.columns.get_loc("target") + 1, "target_sequence",
                  result["target"].map(target_sequences))
    result["has_hit"] = result["target"].notna()

    # Make sure qcov / tcov exist even when there were no hits at all.
    for column in ["pident", "qcov", "tcov"]:
        if column not in result.columns:
            result[column] = np.nan
        result[f"{column}_for_distribution"] = (
            pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        )

    # The only high-similarity criterion
    result["high_similarity"] = (
        result["has_hit"]
        & (result["pident"] >= HIGH_IDENTITY)
        & (result["qcov"] >= HIGH_QUERY_COVERAGE)
        & (result["tcov"] >= HIGH_TARGET_COVERAGE)
    )
    return result


def summarize(activity: str, comparison: str,
              best: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    n = len(best)
    high_n = int(best["high_similarity"].sum())
    summary = {
        "activity": activity,
        "comparison": comparison,
        "n_query": n,
        "n_high_similarity": high_n,
        "high_similarity_percent": 100.0 * high_n / n if n else np.nan,
        "n_no_high_similarity": n - high_n,
        # Distribution stats over the representative hit identity (0 if no hit)
        "median_identity_representative": best["pident_for_distribution"].median(),
        "mean_identity_representative": best["pident_for_distribution"].mean(),
    }
    labels = [
        "Identity ≥ 80 & qcov ≥ 80 & tcov ≥ 80" if row.high_similarity
        else "Below threshold"
        for row in best.itertuples()
    ]
    counts = pd.Series(labels).value_counts().reindex(
        ["Identity ≥ 80 & qcov ≥ 80 & tcov ≥ 80", "Below threshold"], fill_value=0
    )
    intervals = pd.DataFrame({
        "activity": activity,
        "comparison": comparison,
        "category": ["High similarity", "Below threshold"],
        "count": counts.values,
        "percent": 100.0 * counts.values / n if n else np.nan,
    })
    return summary, intervals


def plot_histogram(ax: plt.Axes, best: pd.DataFrame, title: str) -> None:
    """Publication-style histogram: one bar per 5% identity bin, threshold line."""
    bins = np.arange(0, 105, 5)
    other = best.loc[
        best["has_hit"] & ~best["high_similarity"], "pident"
    ].to_numpy()

    high = best.loc[
        best["high_similarity"], "pident"
    ].to_numpy()

    ax.hist(
        [other, high],
        bins=bins,
        stacked=True,
        color=["#3B75AF", "#C0392B"],
        edgecolor="white",
        linewidth=0.8,
        label=[
            "Hits failing one or more thresholds",
            "Identity, qcov and tcov all ≥80%"
        ]
    )
    ax.axvline(HIGH_IDENTITY, color="#C0392B", linestyle="--", linewidth=1.4,
               label=f"{HIGH_IDENTITY:.0f}% identity threshold")

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel("Sequence identity (%)", fontsize=10)
    ax.set_ylabel("Number of query peptides", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")

    total = len(best)
    hit_n = int(best["has_hit"].sum())
    high_n = int(best["high_similarity"].sum())
    high_pct = 100.0 * high_n / total if total else 0.0
    ax.text(
        0.97, 0.95,
        f"Queries: {total}\nHits: {hit_n}\n"
        f"All three thresholds met: {high_n}\n"
        f"({high_pct:.1f}% of all queries)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white",
              "edgecolor": "#888888", "alpha": 0.9},
    )


def save_activity_histograms(activity_results: Dict[str, pd.DataFrame], activity: str,
                             figures_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, comparison in zip(axes, COMPARISONS):
        if comparison in activity_results:
            plot_histogram(ax, activity_results[comparison], comparison.replace("_", " "))
        else:
            ax.axis("off")
            ax.set_title(comparison.replace("_", " "), fontsize=10)
            ax.text(0.5, 0.5, "Unavailable\n(missing dataset)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#666666")
    fig.suptitle(f"{activity}: representative-hit identity distributions", fontsize=14)
    path = figures_dir / f"{activity}_three_comparisons_histograms.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_all_histograms_pdf(all_results: Dict[Tuple[str, str], pd.DataFrame],
                            activities: Iterable[str], path: Path) -> None:
    with PdfPages(path) as pdf:
        for activity in sorted(activities):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
            for ax, comparison in zip(axes, COMPARISONS):
                key = (activity, comparison)
                if key in all_results:
                    plot_histogram(ax, all_results[key], comparison.replace("_", " "))
                else:
                    ax.axis("off")
                    ax.set_title(comparison.replace("_", " "), fontsize=10)
                    ax.text(0.5, 0.5, "Unavailable\n(missing dataset)", ha="center", va="center",
                            transform=ax.transAxes, fontsize=11, color="#666666")
            fig.suptitle(f"{activity}: representative-hit identity distributions", fontsize=14)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def save_heatmap(summary_df: pd.DataFrame, value: str, label: str, path: Path) -> None:
    matrix = summary_df.pivot(index="activity", columns="comparison", values=value)
    matrix = matrix.reindex(columns=list(COMPARISONS))
    height = max(5.0, 0.55 * len(matrix) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, height))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=100,
                linewidths=0.5, cbar_kws={"label": label}, ax=ax)
    ax.set_xlabel("Dataset comparison (query vs target)", fontsize=10)
    ax.set_ylabel("Peptide activity", fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.set_xticklabels([x.get_text().replace("_", "\n") for x in ax.get_xticklabels()],
                       rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def reset_output_directory(output_root: Path) -> None:
    if output_root.exists():
        print(f"Removing previous results: {output_root}", flush=True)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if THREADS < 1 or MAX_TARGET_SEQS < 1:
        raise ValueError("THREADS and MAX_TARGET_SEQS must be positive")
    if shutil.which(BLASTP_EXECUTABLE) is None or shutil.which(MAKEBLASTDB_EXECUTABLE) is None:
        raise FileNotFoundError("Cannot find blastp and/or makeblastdb; check executable paths.")

    activities = validate_dataset_paths()
    output_root = OUTPUT_ROOT.expanduser().resolve()
    reset_output_directory(output_root)

    raw_dir = output_root / "01_blast_raw"
    best_dir = output_root / "02_best_hits"
    high_dir = output_root / "03_high_similarity_sequences"
    summary_dir = output_root / "04_summary"
    figures_dir = output_root / "05_figures"
    tmp_root = output_root / "blast_tmp"
    skipped_dir = summary_dir / "skipped_sequences"
    for directory in [raw_dir, best_dir, high_dir, summary_dir, figures_dir, tmp_root, skipped_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    summaries, interval_tables, length_statistics, skipped_statistics = [], [], [], []
    all_results: Dict[Tuple[str, str], pd.DataFrame] = {}
    print(f"Discovered {len(activities)} activities: {', '.join(activities)}")

    for activity, files in activities.items():
        records_by_file: Dict[str, List[FastaRecord]] = {}
        for name, path in files.items():
            if path is None:
                continue
            result = read_fasta(path)
            records_by_file[name] = result.records
            skipped_statistics.append({
                "activity": activity, "dataset": name, "fasta_path": str(path),
                "total_headers": result.total_headers, "kept": len(result.records),
                "skipped_invalid": len(result.skipped_invalid),
                "skipped_empty": len(result.skipped_empty),
                "skipped_duplicate": len(result.skipped_duplicate),
                "skipped_total": result.total_skipped,
            })
            if result.total_skipped:
                print(f"WARNING: {activity}/{name}: skipped {result.total_skipped} sequences")
                write_skipped_details(result, skipped_dir / f"{activity}__{name}.txt")
            length_statistics.append(calculate_length_statistics(activity, name, path, result.records))

        activity_results: Dict[str, pd.DataFrame] = {}
        for comparison, (query_name, target_name) in COMPARISONS.items():
            if files[query_name] is None or files[target_name] is None:
                missing = [n for n in (query_name, target_name) if files[n] is None]
                print(f"SKIP: {activity} / {comparison}; missing {', '.join(missing)}")
                summaries.append({
                    "activity": activity, "comparison": comparison, "status": "unavailable",
                    "missing_dataset": ", ".join(missing), "n_query": np.nan,
                    "n_high_similarity": np.nan, "high_similarity_percent": np.nan,
                    "n_no_high_similarity": np.nan,
                    "median_identity_representative": np.nan,
                    "mean_identity_representative": np.nan,
                })
                continue

            raw_path = raw_dir / activity / f"{comparison}.tsv"
            tmp_dir = tmp_root / activity / comparison
            run_blast(BLASTP_EXECUTABLE, MAKEBLASTDB_EXECUTABLE,
                      records_by_file[query_name], records_by_file[target_name],
                      raw_path, tmp_dir, THREADS, MAX_TARGET_SEQS)
            raw = load_all_hits(raw_path)
            best = choose_best_hits(raw, records_by_file[query_name], records_by_file[target_name])
            best.insert(0, "activity", activity)
            best.insert(1, "comparison", comparison)

            out_dir = best_dir / activity
            out_dir.mkdir(parents=True, exist_ok=True)
            best.to_csv(out_dir / f"{comparison}_representative_hits.csv", index=False)
            flagged_dir = high_dir / activity
            flagged_dir.mkdir(parents=True, exist_ok=True)
            best.loc[best["high_similarity"]].to_csv(
                flagged_dir / f"{comparison}_identity80_qcov80_tcov80.csv", index=False)

            summary, intervals = summarize(activity, comparison, best)
            summary.update({"status": "completed", "missing_dataset": ""})
            summaries.append(summary)
            interval_tables.append(intervals)
            activity_results[comparison] = best
            all_results[(activity, comparison)] = best

        save_activity_histograms(activity_results, activity, figures_dir)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_dir / "comparison_summary.csv", index=False)
    pd.DataFrame(length_statistics).to_csv(summary_dir / "sequence_length_statistics.csv", index=False)
    if interval_tables:
        pd.concat(interval_tables, ignore_index=True).to_csv(
            summary_dir / "high_similarity_counts.csv", index=False)
    pd.DataFrame(skipped_statistics).to_csv(
        summary_dir / "skipped_sequences_summary.csv", index=False)

    save_heatmap(summary_df, "high_similarity_percent",
                 "High-similarity queries (identity ≥ 80%, qcov ≥ 80%, tcov ≥ 80%)",
                 figures_dir / "heatmap_high_similarity_percent.png")
    save_heatmap(summary_df, "median_identity_representative",
                 "Median representative-hit identity across all queries (no hit = 0%)",
                 figures_dir / "heatmap_median_representative_identity.png")
    save_all_histograms_pdf(all_results, activities.keys(),
                            figures_dir / "all_27_histograms.pdf")

    print(f"\nCompleted {len(all_results)} comparisons. Results: {output_root}")
    print("Primary table:", summary_dir / "comparison_summary.csv")
    print("Primary heatmap:", figures_dir / "heatmap_high_similarity_percent.png")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)