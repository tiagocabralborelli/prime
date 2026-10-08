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


from pathlib import Path
import yaml

config_path = Path(__file__).with_name("Params.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

train_config = config["train"]


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


def append_bootstrap_metrics(output_dir, name, bootstrap_results):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ('auc', 'ap', 'accuracy', 'precision', 'recall', 'f1', 'mcc')
    columns = ['model_name'] + [
        f'{metric}_{stat}'
        for metric in metrics
        for stat in ('mean', 'std')
    ]
    row = {'model_name': name}
    for metric in metrics:
        values = bootstrap_results.get(metric, {})
        row[f'{metric}_mean'] = values.get('mean', '')
        row[f'{metric}_std'] = values.get('std', '')

    table_path = os.path.join(output_dir, 'bootstrap_metrics.csv')
    has_content = os.path.isfile(table_path) and os.path.getsize(table_path) > 0
    if has_content:
        with open(table_path, 'r', newline='', encoding='utf-8') as table_file:
            existing_columns = next(csv.reader(table_file), [])
        if existing_columns != columns:
            raise ValueError(
                f"Existing bootstrap table has unexpected columns: {table_path}"
            )

    with open(table_path, 'a', newline='', encoding='utf-8') as table_file:
        writer = csv.DictWriter(table_file, fieldnames=columns)
        if not has_content:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended bootstrap metrics to {table_path}")


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

def main(train, test, name, base_model, seed=train_config["seed"], smote=False,
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
    model_path = os.path.join(model_dir, f'{name}_model.pkl')
    
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
        'roc_auc': float(roc_auc_score(test_labels, test_proba)),
        'average_precision': float(average_precision_score(test_labels, test_proba)),
        'accuracy': float(accuracy_score(test_labels, test_pred)),
        'precision': float(precision_score(test_labels, test_pred, zero_division=0)),
        'recall': float(recall_score(test_labels, test_pred, zero_division=0)),
        'f1': float(f1_score(test_labels, test_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(test_labels, test_pred)),
        'confusion_matrix': confusion_matrix(test_labels, test_pred).tolist(),
    }
    print({'test_metrics': metrics})
    bootstrap_results = bootstrap_metrics(test_labels, test_pred, test_proba)
    print({'test_bootstrap_95_ci_and_std': bootstrap_results})
    append_bootstrap_metrics(output_dir, name, bootstrap_results)
    plot_dir = os.path.join(output_dir, name)
    plot_binary_evaluation(test_labels, test_pred, test_proba, plot_dir, name)
    print(f"Saved evaluation plots to {plot_dir}")


if __name__ == "__main__":
    main()