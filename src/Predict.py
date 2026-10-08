import csv
import os
import pickle

import click
import numpy as np
import torch
from Bio import SeqIO
from transformers import EsmForMaskedLM, EsmTokenizer

from Embedder.Extractor import EmbedderLoRAESM
from Loader.Dataset import filter_sequence


def load_artifact(model_path):
    # Só carregue pickles que você confia.
    with open(model_path, "rb") as model_file:
        artifact = pickle.load(model_file)
    if artifact.get("artifact_version") != 1:
        raise ValueError(f"Artefato incompatível: {model_path}")

    required_fields = (
        "model",
        "scaler",
        "threshold",
        "classes",
        "positive_label",
        "embedding",
    )
    missing_fields = [
        field for field in required_fields if field not in artifact
    ]
    if missing_fields:
        raise ValueError(f"Campos ausentes em {model_path}: {missing_fields}")

    checkpoint = artifact["embedding"].get("checkpoint")
    if not checkpoint:
        raise ValueError(f"O artefato não informa checkpoint: {model_path}")
    return artifact, checkpoint


def predict_with_artifact(artifact, embeddings):
    scaled_embeddings = artifact["scaler"].transform(
        np.asarray(embeddings, dtype=float)
    )

    positive_label = artifact["positive_label"]
    classes = np.asarray(artifact["classes"])
    if len(classes) != 2 or not np.any(classes == positive_label):
        raise ValueError("O artefato não contém duas classes e classe positiva válida.")
    negative_label = classes[classes != positive_label][0]

    classifiers = artifact.get("models") or {"selected": artifact["model"]}
    probabilities_by_classifier = []
    for classifier in classifiers.values():
        classifier_classes = np.asarray(classifier.classes_)
        positive_indices = np.flatnonzero(classifier_classes == positive_label)
        if len(positive_indices) != 1:
            raise ValueError("Classe positiva não encontrada unicamente no classificador.")
        probabilities_by_classifier.append(
            classifier.predict_proba(scaled_embeddings)[:, int(positive_indices[0])]
        )

    probabilities = np.mean(probabilities_by_classifier, axis=0)
    threshold = float(artifact["threshold"])
    predictions = np.where(
        probabilities >= threshold, positive_label, negative_label
    )
    return predictions, probabilities, threshold


def weighted_sigmoid_score(probabilities_by_model, model_weights):
    model_names = set(probabilities_by_model)
    weight_names = set(model_weights)
    if model_names != weight_names:
        missing_weights = sorted(model_names - weight_names)
        unknown_weights = sorted(weight_names - model_names)
        raise ValueError(
            f"Pesos precisam corresponder aos modelos carregados. "
            f"Sem peso: {missing_weights}; modelos sem artefato: {unknown_weights}."
        )

    if not probabilities_by_model:
        raise ValueError("Nenhuma probabilidade de modelo foi fornecida.")

    weighted_sum = None
    expected_shape = None
    for model_name, probabilities in probabilities_by_model.items():
        probabilities = np.asarray(probabilities, dtype=float)
        weight = float(model_weights[model_name])
        if not np.isfinite(weight):
            raise ValueError(f"Peso inválido para {model_name}: {weight}")
        if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)):
            raise ValueError(f"Probabilidades inválidas para o modelo {model_name}.")
        if expected_shape is None:
            expected_shape = probabilities.shape
            weighted_sum = np.zeros(expected_shape, dtype=float)
        elif probabilities.shape != expected_shape:
            raise ValueError("Todos os modelos precisam prever as mesmas proteínas.")
        weighted_sum += weight * probabilities

    return 1.0 / (1.0 + np.exp(-np.clip(weighted_sum, -709.0, 709.0)))


@click.command()
@click.option(
    "--models",
    "model_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Arquivo .pkl criado pelo prime-train; repita a opção para cada modelo",
)
@click.option(
    "--input",
    "input_fasta",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="FASTA com sequências para classificar",
)
@click.option(
    "--output",
    "output_csv",
    required=True,
    type=click.Path(dir_okay=False),
    help="CSV onde as predições serão salvas",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
    help="Dispositivo para gerar embeddings",
)
def main(model_paths, input_fasta, output_csv, device):
    artifacts = []
    for model_path in model_paths:
        try:
            artifact, checkpoint = load_artifact(model_path)
        except (KeyError, TypeError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        model_name = artifact.get("name") or os.path.splitext(
            os.path.basename(model_path)
        )[0]
        artifacts.append((model_name, model_path, artifact, checkpoint))

    records = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        sequence = str(record.seq).upper()
        if not sequence or not filter_sequence(sequence):
            click.echo(
                f"Aviso: sequência ignorada por aminoácido inválido: {record.id}",
                err=True,
            )
            continue
        records.append((record.id, sequence))

    if not records:
        raise click.ClickException(
            "Nenhuma sequência válida foi encontrada no FASTA."
        )

    sequences = [sequence for _, sequence in records]
    embeddings_by_checkpoint = {}
    for _, _, _, checkpoint in artifacts:
        if checkpoint in embeddings_by_checkpoint:
            continue
        extractor = EmbedderLoRAESM(path=checkpoint, device=device)
        extractor.loadModel(
            tokenizer=EsmTokenizer,
            model_loader=EsmForMaskedLM,
        )
        embeddings_by_checkpoint[checkpoint] = extractor.embedding(sequences)
        del extractor

    output_directory = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_directory, exist_ok=True)



    model_weights = {
        "Antimicrobial":1.0,
        "Hemolytic":1.0,
        "Toxicity":1.0
    }

    individual_predictions = {}
    probabilities_by_model = {}
    thresholds_by_model = {}
    for model_name, _, artifact, checkpoint in artifacts:
        if model_name in probabilities_by_model:
            raise click.ClickException(
                f"Nome de modelo duplicado nos artefatos: {model_name}"
            )
        try:
            predictions, probabilities, threshold = predict_with_artifact(
                artifact, embeddings_by_checkpoint[checkpoint]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise click.ClickException(
                f"Falha no modelo {model_name}: {error}"
            ) from error
        individual_predictions[model_name] = predictions
        probabilities_by_model[model_name] = probabilities
        thresholds_by_model[model_name] = threshold

    try:
        combined_scores = weighted_sigmoid_score(
            probabilities_by_model, model_weights
        )
    except (TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "model_name",
                "sequence_id",
                "sequence",
                "predicted_label",
                "positive_probability",
                "threshold",
                "weighted_sigmoid_score",
            )
        )
        for model_name, predictions in individual_predictions.items():
            probabilities = probabilities_by_model[model_name]
            threshold = thresholds_by_model[model_name]
            for (record_id, sequence), prediction, probability, combined_score in zip(
                records, predictions, probabilities, combined_scores
            ):
                writer.writerow(
                    (
                        model_name,
                        record_id,
                        sequence,
                        prediction.item(),
                        float(probability),
                        threshold,
                        float(combined_score),
                    )
                )

    click.echo(
        f"Predições salvas em {output_csv}: {len(artifacts)} modelos x "
        f"{len(records)} proteínas = {len(artifacts) * len(records)} linhas."
    )


if __name__ == "__main__":
    main()