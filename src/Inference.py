#!/usr/bin/env python3
import csv
import os
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
    """Retorna o nome da feature a partir do nome do arquivo .pkl (ex: Antimicrobial_... -> Antimicrobial)."""
    return os.path.basename(filename).split("_")[0]


def run_inference_on_model(artifact, embeddings, device):
    """Executa a predição para um artefato carregado."""
    scaler = artifact["scaler"]
    scaled_embeddings = scaler.transform(embeddings)

    model = artifact["model"]
    if hasattr(model, "model_"):
        model.model_.to(device)
        model.device = str(device)

    positive_label = artifact["positive_label"]
    classes = np.asarray(model.classes_)
    pos_idx = int(np.flatnonzero(classes == positive_label)[0])

    probs = model.predict_proba(scaled_embeddings)[:, pos_idx]
    threshold = float(artifact.get("threshold", 0.5))
    return probs, threshold


@click.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Caminho do arquivo FASTA de entrada.",
)
@click.option(
    "--model_dir",
    "--model-dir",
    "model_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Pasta onde os arquivos .pkl dos modelos estão localizados.",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    default="predictions.csv",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Caminho do arquivo CSV de saída.",
)
@click.option(
    "--device",
    "-d",
    "device",
    default="cpu",
    show_default=True,
    type=click.Choice(["cpu", "cuda", "gpu"], case_sensitive=False),
    help="Dispositivo de execução ('cpu' ou 'cuda'/'gpu').",
)
@click.option(
    "--checkpoint",
    "-c",
    "checkpoint",
    default="/data/tiago/prime-validation/checkpoint-500000",
    show_default=True,
    help="Caminho do checkpoint do modelo ESM.",
)
@click.option(
    "--include_pred",
    "--include-pred",
    "include_pred",
    is_flag=True,
    default=False,
    help="Inclui coluna de predição binária (0/1) no CSV.",
)
@click.option(
    "--include_std",
    "--include-std",
    "include_std",
    is_flag=True,
    default=False,
    help="Inclui coluna de desvio padrão (incerteza) no CSV.",
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

    # Garante que pickles criados no CUDA possam ser lidos em CPU sem erro
    if device_name == "cpu":
        torch.serialization.default_restore_location = lambda storage, loc: storage

    # 1. Leitura do FASTA
    records = []
    for record in SeqIO.parse(input_file, "fasta"):
        seq = str(record.seq).upper()
        if seq and filter_sequence(seq):
            records.append((str(record.id), seq))

    if not records:
        raise click.ClickException("Nenhuma sequência válida encontrada no arquivo FASTA.")

    sequences = [seq for _, seq in records]
    print(f"Carregados {len(records)} peptídeos de {input_file}.")

    # 2. Extração dos embeddings na memória via ESM
    print(f"Extraindo embeddings com ESM de {checkpoint} ({device_name})...")
    extractor = EmbedderLoRAESM(path=checkpoint, device=device_name)
    extractor.loadModel(tokenizer=EsmTokenizer, model_loader=EsmForMaskedLM)
    embeddings = extractor.embedding(sequences)
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Agrupamento dos modelos .pkl por feature
    model_paths = sorted(Path(model_dir).glob("*.pkl"))
    if not model_paths:
        raise click.ClickException(f"Nenhum arquivo .pkl encontrado em {model_dir}")

    feature_models = defaultdict(list)
    for path in model_paths:
        feature_name = get_feature_name(path.name)
        feature_models[feature_name].append(path)

    features = sorted(feature_models.keys())
    print(f"Encontradas {len(features)} features em {model_dir}: {', '.join(features)}")

    # 4. Inferência e cálculo da média para cada feature
    results = {}
    for feature in features:
        paths = feature_models[feature]
        all_probs = []
        all_thresholds = []

        for pkl_file in paths:
            with open(pkl_file, "rb") as f:
                import pickle
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

    # 5. Exportação para CSV
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

    print(f"Concluído! Predições salvas em {output_file}.")


if __name__ == "__main__":
    main()
