import click 
from Loader.Dataset import DataSet
from Embedder.Extractor import EmbedderESM, EmbedderLoRAESM
from Coach.Classifiers import Coach
from Metrics.Calculator import bootstrap_metrics
from Metrics.Calculator import plot_binary_evaluation
import os
import pickle
import hashlib
import csv
import json
import platform
from importlib.metadata import PackageNotFoundError, version

import torch
from transformers import EsmTokenizer, EsmForMaskedLM
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np

def select_best_classifier(cv_models,cv_scores, mode):
    if mode == "auto":
        best_name = max(cv_scores, key=cv_scores.get)

        final_models = {best_name: cv_models[best_name]}
    elif mode == 'both':
        final_models = cv_models
        best_name = 'both'
    else:
        best_name = mode
        final_models = {best_name: cv_models[best_name]}
    return final_models

def select_threshold_cv_binary(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = np.argmax(f1)
    best_thresh = thr[best_idx]
    return best_thresh


def get_embeddings_with_cache(sequences, extractor, cache_file):
    cache_identity = f"{extractor.check_point}\n" + '\n'.join(sequences)
    sequence_hash = hashlib.sha256(cache_identity.encode('utf-8')).hexdigest()
    if os.path.exists(cache_file):
        with np.load(cache_file) as cached:
            if 'sequence_hash' in cached and str(cached['sequence_hash']) == sequence_hash:
                print(f"Loading cached embeddings from {cache_file}")
                return cached['embeddings']

    embeddings = np.asarray(extractor.embedding(sequences), dtype=float)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, embeddings=embeddings, sequence_hash=sequence_hash)
    print(f"Saved embeddings to {cache_file}")
    return embeddings


def upsert_csv_rows(output_dir, filename, columns, rows, key_columns,
                    replace_scope_columns=None):
    os.makedirs(output_dir, exist_ok=True)
    table_path = os.path.join(output_dir, filename)
    has_content = os.path.isfile(table_path) and os.path.getsize(table_path) > 0
    stored_rows = {}
    if has_content:
        with open(table_path, 'r', newline='', encoding='utf-8') as table_file:
            reader = csv.DictReader(table_file)
            existing_columns = reader.fieldnames or []
            if existing_columns != columns:
                raise ValueError(
                    f'Existing table has unexpected columns: {table_path}'
                )
            for existing_row in reader:
                key = tuple(existing_row[column] for column in key_columns)
                stored_rows[key] = existing_row

    if replace_scope_columns and rows:
        replace_scopes = {
            tuple(str(row[column]) for column in replace_scope_columns)
            for row in rows
        }
        for key, existing_row in list(stored_rows.items()):
            scope = tuple(
                existing_row[column] for column in replace_scope_columns
            )
            if scope in replace_scopes:
                del stored_rows[key]

    for row in rows:
        key = tuple(str(row[column]) for column in key_columns)
        stored_rows[key] = row

    with open(table_path, 'w', newline='', encoding='utf-8') as table_file:
        writer = csv.DictWriter(table_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(stored_rows.values())
    print(f'Saved results to {table_path}')


def append_bootstrap_metrics(output_dir, name, seed, bootstrap_results):
    metrics = ('auc', 'ap', 'accuracy', 'precision', 'recall', 'f1', 'mcc')
    columns = ['model_name', 'seed'] + [
        f'{metric}_{stat}'
        for metric in metrics
        for stat in ('mean', 'std', 'ci_low', 'ci_high')
    ]
    row = {'model_name': name, 'seed': seed}
    for metric in metrics:
        values = bootstrap_results.get(metric, {})
        for stat in ('mean', 'std', 'ci_low', 'ci_high'):
            row[f'{metric}_{stat}'] = values.get(stat, '')

    upsert_csv_rows(
        output_dir,
        'bootstrap_metrics_by_seed.csv',
        columns,
        [row],
        key_columns=('model_name', 'seed'),
    )


def append_run_outputs(output_dir, name, seed, selected_models, model_params,
                       cv_scores, threshold, train_size, test_sequences,
                       test_labels, test_pred, test_proba, metrics):
    metric_names = (
        'roc_auc', 'average_precision', 'accuracy', 'precision', 'recall',
        'f1', 'mcc',
    )
    summary_columns = [
        'model_name', 'seed', 'selected_models', 'model_parameters',
        'cv_scores', 'threshold', 'train_size', 'test_size', *metric_names,
        'confusion_matrix',
    ]
    summary_row = {
        'model_name': name,
        'seed': seed,
        'selected_models': json.dumps(selected_models),
        'model_parameters': json.dumps(model_params, default=str),
        'cv_scores': json.dumps(cv_scores),
        'threshold': float(threshold),
        'train_size': train_size,
        'test_size': len(test_labels),
        'confusion_matrix': json.dumps(metrics['confusion_matrix']),
    }
    summary_row.update({metric: metrics[metric] for metric in metric_names})
    upsert_csv_rows(
        output_dir,
        'seed_run_summary.csv',
        summary_columns,
        [summary_row],
        key_columns=('model_name', 'seed'),
    )

    prediction_columns = [
        'model_name', 'seed', 'sample_index', 'sequence', 'true_label',
        'predicted_label', 'positive_probability', 'threshold',
    ]
    prediction_rows = [
        {
            'model_name': name,
            'seed': seed,
            'sample_index': sample_index,
            'sequence': sequence,
            'true_label': int(label),
            'predicted_label': int(prediction),
            'positive_probability': float(probability),
            'threshold': float(threshold),
        }
        for sample_index, (sequence, label, prediction, probability) in enumerate(
            zip(test_sequences, test_labels, test_pred, test_proba)
        )
    ]
    upsert_csv_rows(
        output_dir,
        'seed_predictions.csv',
        prediction_columns,
        prediction_rows,
        key_columns=('model_name', 'seed', 'sample_index'),
        replace_scope_columns=('model_name', 'seed'),
    )


def aggregate_seed_outputs(output_dir, name):
    summary_path = os.path.join(output_dir, 'seed_run_summary.csv')
    with open(summary_path, 'r', newline='', encoding='utf-8') as summary_file:
        summary_rows = [
            row for row in csv.DictReader(summary_file)
            if row['model_name'] == name
        ]

    metric_names = (
        'roc_auc', 'average_precision', 'accuracy', 'precision', 'recall',
        'f1', 'mcc', 'threshold',
    )
    aggregate_row = {
        'model_name': name,
        'n_seeds': len(summary_rows),
        'selected_models': json.dumps(sorted({
            model
            for row in summary_rows
            for model in json.loads(row['selected_models'])
        })),
    }
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in summary_rows])
        aggregate_row[f'{metric}_mean'] = float(np.mean(values))
        aggregate_row[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    aggregate_columns = [
        'model_name', 'n_seeds', 'selected_models',
        *[f'{metric}_{stat}' for metric in metric_names for stat in ('mean', 'std')],
    ]
    upsert_csv_rows(
        output_dir,
        'seed_aggregate_metrics.csv',
        aggregate_columns,
        [aggregate_row],
        key_columns=('model_name',),
    )

    predictions_path = os.path.join(output_dir, 'seed_predictions.csv')
    with open(predictions_path, 'r', newline='', encoding='utf-8') as predictions_file:
        prediction_rows = [
            row for row in csv.DictReader(predictions_file)
            if row['model_name'] == name
        ]

    grouped_predictions = {}
    for row in prediction_rows:
        grouped_predictions.setdefault(row['sample_index'], []).append(row)

    uncertainty_rows = []
    for sample_index, rows in sorted(
        grouped_predictions.items(), key=lambda item: int(item[0])
    ):
        first_row = rows[0]
        if any(
            row['sequence'] != first_row['sequence']
            or row['true_label'] != first_row['true_label']
            for row in rows
        ):
            raise ValueError(
                'Test sequences or labels differ between seed runs; '
                'per-sequence uncertainty cannot be aligned.'
            )
        probabilities = np.asarray(
            [float(row['positive_probability']) for row in rows]
        )
        positive_rate = float(np.mean([
            int(row['predicted_label']) for row in rows
        ]))
        uncertainty_rows.append({
            'model_name': name,
            'sample_index': sample_index,
            'sequence': first_row['sequence'],
            'true_label': int(first_row['true_label']),
            'n_seeds': len(rows),
            'probability_mean': float(np.mean(probabilities)),
            'probability_std': float(np.std(probabilities, ddof=1)) if len(probabilities) > 1 else 0.0,
            'positive_prediction_rate': positive_rate,
            'prediction_disagreement_rate': min(positive_rate, 1.0 - positive_rate),
        })

    uncertainty_columns = [
        'model_name', 'sample_index', 'sequence', 'true_label', 'n_seeds',
        'probability_mean', 'probability_std', 'positive_prediction_rate',
        'prediction_disagreement_rate',
    ]
    upsert_csv_rows(
        output_dir,
        'seed_prediction_uncertainty.csv',
        uncertainty_columns,
        uncertainty_rows,
        key_columns=('model_name', 'sample_index'),
        replace_scope_columns=('model_name',),
    )


def get_package_versions():
    packages = ('numpy', 'scikit-learn', 'torch', 'transformers', 'peft')
    versions = {'python': platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


@click.command()
@click.option('--train', required = True, help = "Training FASTA file with sequence labels in the FASTA IDs")
@click.option('--test', required = True,  help = "Test FASTA file with sequence labels in the FASTA IDs")
@click.option('--name', required = True,  help = "Project name")
@click.option('--base-model', required = True, help = "Pre-trained model folder")
@click.option('--seed', type=int, default=42, show_default=True, help="Random state")
@click.option('--smote', is_flag=True, default=False, help="Enable SMOTE")
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
              show_default=True, help='Device used for embedding extraction')
@click.option('--model-dir', default='saved_models', show_default=True,
              help='Directory where the deployable model artifact is saved')
@click.option('--embedding-cache-dir', default='embedding_cache', show_default=True,
              help='Directory to cache raw train/test embeddings')
@click.option('--output-dir', default='results', show_default=True,
              help='Directory for evaluation plots')

def main(train, test, name, base_model, seed=42, smote=False,
         device='cuda' if torch.cuda.is_available() else 'cpu', model_dir='saved_models',
         embedding_cache_dir='embedding_cache', output_dir='results'):
    #Hard-coded parameters
    
    HEADS = {
    "linear": {
        "lr": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "param_grid":{'C': [0.01, 0.1, 1.0, 10.0]},
        "n_jobs":26
    },
    "non-linear":{
        "mlp":MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', alpha=0.001, batch_size=32, learning_rate='adaptive', max_iter=100, early_stopping=True, validation_fraction=0.2),
        "param_grid":{
            "binary": {
                'hidden_layer_sizes': [(32,), (64,), (128,), (64,32)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]},
            "multiclass":{
                'hidden_layer_sizes': [(32,), (64,), (128,), (64,32)],
                'alpha': [0.0001, 0.001, 0.01]}
                    },
        "n_jobs":26
    }
}

    #Load Data
    DataObj = DataSet(
        train = train,
        test = test,
        name = name)
    DataObj.load(sample = False)
    DataObj.clean()

    if not DataObj.train_seq:
        raise click.ClickException('No training sequences remain after filtering and cleaning.')
    if not DataObj.test_seq:
        raise click.ClickException('No test sequences remain after filtering and contamination removal.')
    if set(np.unique(DataObj.train_id).tolist()) != {0.0, 1.0}:
        raise click.ClickException('Binary training labels must contain both 0 and 1.')
    if set(np.unique(DataObj.test_id).tolist()) != {0.0, 1.0}:
        raise click.ClickException('Binary test labels must contain both 0 and 1 for ROC-AUC evaluation.')


    print("Try LoRA")
    ExtratorESM = EmbedderLoRAESM(
        path = base_model,
        device = device
    )
    ExtratorESM.loadModel(tokenizer=EsmTokenizer, model_loader = EsmForMaskedLM)

    #Train classifiers
    train_cache = os.path.join(embedding_cache_dir, f'{name}_train_emb.npz')
    test_cache = os.path.join(embedding_cache_dir, f'{name}_test_emb.npz')
    train_embedded = get_embeddings_with_cache(DataObj.train_seq, ExtratorESM, train_cache)
    train_lables = DataObj.train_id
    ModelSelectorObj = Coach(
        train_emb = train_embedded,
        train_labels = train_lables,
        seed = seed)
    ModelSelectorObj.preprocess()
    ModelSelectorObj.select_binary_classifier(model_dict=HEADS, classifier_choice='auto', use_smote = smote)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{name}_seed_{seed}_model.pkl')
    
    artifact = {
        'artifact_version': 1,
        'task_type': 'binary_classification',
        'model': ModelSelectorObj.best_model_obj,
        'models': ModelSelectorObj.trained_best_models if len(ModelSelectorObj.trained_best_models) > 1 else None,
        'threshold': float(ModelSelectorObj.threshold),
        'scaler': ModelSelectorObj.scaler,
        'seed': seed,
        'name': name,
        'classifier_names': list(ModelSelectorObj.trained_best_models),
        'classes': ModelSelectorObj.best_model_obj.classes_.tolist(),
        'positive_label': 1.0,
        'use_embeddings': True,
        'embedding': {
            'checkpoint': base_model,
            'tokenizer': EsmTokenizer.__name__,
            'model_loader': EsmForMaskedLM.__name__,
            'pooling': 'attention-mask mean of the last hidden layer',
            'max_length': int(ExtratorESM.max_len),
        },
        'package_versions': get_package_versions(),
    }
    temporary_model_path = f'{model_path}.{os.getpid()}.tmp'
    with open(temporary_model_path, 'wb') as model_file:
        pickle.dump(artifact, model_file)
    os.replace(temporary_model_path, model_path)
    print(f"Saved deployable model artifact to {model_path}")

    test_pred, test_proba = ModelSelectorObj.predict(
        test_emb=get_embeddings_with_cache(DataObj.test_seq, ExtratorESM, test_cache),
        test_labels=DataObj.test_id)

    test_labels = np.asarray(DataObj.test_id)
    metrics = {
        'seed': seed,
        'roc_auc': float(roc_auc_score(test_labels, test_proba)),
        'average_precision': float(average_precision_score(test_labels, test_proba)),
        'accuracy': float(accuracy_score(test_labels, test_pred)),
        'precision': float(precision_score(test_labels, test_pred, zero_division=0)),
        'recall': float(recall_score(test_labels, test_pred, zero_division=0)),
        'f1': float(f1_score(test_labels, test_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(test_labels, test_pred)),
        'confusion_matrix': confusion_matrix(test_labels, test_pred).tolist(),
    }
    print({'seed': seed, 'test_metrics': metrics})
    bootstrap_results = bootstrap_metrics(test_labels, test_pred, test_proba)
    print({'seed': seed, 'test_bootstrap_95_ci_and_std': bootstrap_results})
    append_bootstrap_metrics(output_dir, name, seed, bootstrap_results)
    selected_models = list(ModelSelectorObj.trained_best_models)
    model_params = {
        model_name: model.get_params(deep=False)
        for model_name, model in ModelSelectorObj.trained_best_models.items()
    }
    cv_scores = {
        model_name: float(score)
        for model_name, score in ModelSelectorObj.cross_val_scores.items()
    }
    append_run_outputs(
        output_dir=output_dir,
        name=name,
        seed=seed,
        selected_models=selected_models,
        model_params=model_params,
        cv_scores=cv_scores,
        threshold=float(ModelSelectorObj.threshold),
        train_size=len(DataObj.train_seq),
        test_sequences=DataObj.test_seq,
        test_labels=test_labels,
        test_pred=test_pred,
        test_proba=test_proba,
        metrics=metrics,
    )
    aggregate_seed_outputs(output_dir, name)
    plot_dir = os.path.join(output_dir, name, f'seed_{seed}')
    plot_name = f'{name}_seed_{seed}'
    plot_binary_evaluation(test_labels, test_pred, test_proba, plot_dir, plot_name)
    print(f"Saved evaluation plots to {plot_dir}")


if __name__ == "__main__":
    main()