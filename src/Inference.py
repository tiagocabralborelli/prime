#!/usr/bin/env python3
"""
Inference.py - Multi-Model Ensemble CLI for Peptide Property Prediction.

Loads 15 trained models per feature (e.g., Allergen, Antimicrobial, Fouling, Hemolitic, Toxicity)
and calculates the ensemble mean probability (and optional uncertainty / binary prediction)
for each peptide in the input FASTA file.
"""

import csv
import os
import pickle
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import torch
from Bio import SeqIO
from transformers import EsmForMaskedLM, EsmTokenizer

from Embedder.Extractor import EmbedderLoRAESM
from Loader.Dataset import filter_sequence


def get_feature_name(filename):
    """Returns the feature name extracted from the .pkl model filename (e.g. Antimicrobial_... -> Antimicrobial)."""
    return os.path.basename(filename).split("_")[0]


def run_inference_on_model(artifact, embeddings, device):
    """Runs inference for a loaded model artifact."""
    scaler = artifact["scaler"]
    scaled_embeddings = scaler.transform(embeddings)

    model = artifact["model"]
    if hasattr(model, "model_"):
        model.model_.to(device)
        model.device = str(device)

    positive_label = artifact["positive_label"]
    classes = np.asarray(model.classes_)
    pos_indices = np.flatnonzero(classes == positive_label)
    if len(pos_indices) != 1:
        raise ValueError("Positive label not uniquely found in classifier classes.")
    pos_idx = int(pos_indices[0])

    probs = model.predict_proba(scaled_embeddings)[:, pos_idx]
    threshold = float(artifact.get("threshold", 0.5))
    return probs, threshold


@click.command(help="PRIME Multi-Model Ensemble Inference CLI.")
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to input FASTA file containing peptides to analyze.",
)
@click.option(
    "--model_dir",
    "--model-dir",
    "model_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the trained .pkl model files.",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    default="predictions.csv",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to output CSV file for saving predictions.",
)
@click.option(
    "--device",
    "-d",
    "device",
    default="cpu",
    show_default=True,
    type=click.Choice(["cpu", "cuda", "gpu"], case_sensitive=False),
    help="Execution device ('cpu' or 'cuda'/'gpu').",
)
@click.option(
    "--checkpoint",
    "-c",
    "checkpoint",
    default="/data/tiago/prime-validation/checkpoint-500000",
    show_default=True,
    help="Path to base ESM model checkpoint directory.",
)
@click.option(
    "--include_pred",
    "--include-pred",
    "include_pred",
    is_flag=True,
    default=False,
    help="Include binary prediction column (0/1) based on average threshold.",
)
@click.option(
    "--include_std",
    "--include-std",
    "include_std",
    is_flag=True,
    default=False,
    help="Include standard deviation column (epistemic uncertainty across seeds).",
)
def main(
    input_file,
    model_dir,
    output_file,
    device,
    checkpoint,
    include_pred,
    include_std,
):
    device_name = "cuda" if device.lower() in ["cuda", "gpu"] and torch.cuda.is_available() else "cpu"
    target_device = torch.device(device_name)

    # Ensure CUDA-saved models can be unpickled cleanly on CPU
    if device_name == "cpu":
        torch.serialization.default_restore_location = lambda storage, loc: storage

    click.echo("=" * 60)
    click.echo("PRIME Multi-Model Ensemble Inference CLI")
    click.echo("=" * 60)
    click.echo(f"• Input FASTA: {input_file}")
    click.echo(f"• Models directory: {model_dir}")
    click.echo(f"• Output CSV: {output_file}")
    click.echo(f"• Checkpoint: {checkpoint}")
    click.echo(f"• Device: {device_name}")
    click.echo("=" * 60)

    # 1. Parse and filter FASTA sequences
    records = []
    skipped = 0
    for record in SeqIO.parse(input_file, "fasta"):
        seq = str(record.seq).upper()
        if seq and filter_sequence(seq):
            records.append((str(record.id), seq))
        else:
            skipped += 1

    if not records:
        raise click.ClickException("No valid peptide sequences found in FASTA file.")

    sequences = [seq for _, seq in records]
    click.echo(f"Loaded {len(records)} valid peptides from {input_file} ({skipped} skipped).")

    # 2. Extract embeddings in-memory using ESM
    click.echo(f"Extracting embeddings using ESM from {checkpoint} ({device_name})...")
    extractor = EmbedderLoRAESM(path=checkpoint, device=device_name)
    extractor.loadModel(tokenizer=EsmTokenizer, model_loader=EsmForMaskedLM)
    embeddings = extractor.embedding(sequences)
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    click.echo(f"Embeddings extracted: shape {embeddings.shape}")

    # 3. Group models by feature
    model_paths = sorted(Path(model_dir).glob("*.pkl"))
    if not model_paths:
        raise click.ClickException(f"No .pkl model files found in {model_dir}")

    feature_models = defaultdict(list)
    for path in model_paths:
        feature_name = get_feature_name(path.name)
        feature_models[feature_name].append(path)

    features = sorted(feature_models.keys())
    click.echo(f"Found {len(features)} target features in {model_dir}: {', '.join(features)}")

    # 4. Run inference and aggregate per feature
    results = {}
    for feature in features:
        paths = feature_models[feature]
        click.echo(f"Processing feature '{feature}' ({len(paths)} models)...")
        all_probs = []
        all_thresholds = []

        for pkl_file in paths:
            with open(pkl_file, "rb") as f:
                artifact = pickle.load(f)

            probs, threshold = run_inference_on_model(artifact, embeddings, target_device)
            all_probs.append(probs)
            all_thresholds.append(threshold)

        all_probs = np.array(all_probs)  # shape: (n_models, n_samples)
        mean_probs = np.mean(all_probs, axis=0)

        results[feature] = {
            "mean_prob": mean_probs,
            "std_prob": np.std(all_probs, axis=0),
            "pred": (mean_probs >= np.mean(all_thresholds)).astype(int),
        }

    # 5. Export results to CSV
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    header = ["sequence_id", "sequence"]
    for feature in features:
        header.append(f"{feature}_prob_mean")
        if include_std:
            header.append(f"{feature}_prob_std")
        if include_pred:
            header.append(f"{feature}_pred")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, (seq_id, seq) in enumerate(records):
            row = [seq_id, seq]
            for feature in features:
                row.append(f"{results[feature]['mean_prob'][i]:.5f}")
                if include_std:
                    row.append(f"{results[feature]['std_prob'][i]:.5f}")
                if include_pred:
                    row.append(str(results[feature]['pred'][i]))
            writer.writerow(row)

    click.echo("=" * 60)
    click.echo(f"✓ Inference completed successfully! Predictions saved to: {output_file}")
    click.echo(f"  Processed {len(records)} peptides across {len(features)} features.")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
