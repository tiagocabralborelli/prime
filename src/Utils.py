#!/usr/bin/env python3
"""
Multi‑Property Predictor – ESM‑2 Embedding Based (SMOTE + SHAP)
=================================================================
- SMOTE for binary classification (optional, use --use_smote)
- SHAP summary plots saved for logistic regression models
- Full DNN grid, no dataset size reduction
- Thread‑safe (MLP n_jobs=32)
"""

import os
import sys
import argparse
import pickle
import warnings
from datetime import datetime
from copy import deepcopy

# Force non‑interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as stats
from Bio import SeqIO
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
    confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE
from Metrics.Calculator import bootstrap_metrics

import logging
def ConfigureLogging(FileName):
    """
    Basic log settings
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    
    # Evita duplicação de handlers se rodar mais de uma vez
    if not logger.handlers:
        file_handler = logging.FileHandler(FileName, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)



# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ESM‑2 imports
try:
    import torch
    from transformers import EsmTokenizer, EsmForMaskedLM
    from peft import PeftModel
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

warnings.filterwarnings('ignore')

# Allowed amino acids
ALLOWED_AA = set('ACDEFGHIKLMNPQRSTVWY')


def parse_uncertainty(uncertainty_dict):
    grouped_values = defaultdict(lambda: defaultdict(list))

    for seed, properties in uncertainty_dict.items():
        for prop_name, metrics in properties.items():
            for metric_name, value in metrics.items():
                grouped_values[prop_name][metric_name].append(value)
    stats_results = {}
    for prop_name, metrics in grouped_values.items():
        stats_results[prop_name] = {}
        for metric_name, values in metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            stats_results[prop_name][metric_name] = {'mean': mean, 'std': std}
            logging.info(f"  {metric_name.upper():<10}: {mean:.4f} ± {std:.9f}")
    return stats_results

def filter_sequence(seq):
    seq_upper = seq.upper()
    for ch in seq_upper:
        if ch not in ALLOWED_AA:
            return False
    return True

def load_fasta_labels(filepath):
    seqs, labels = [], []
    for rec in SeqIO.parse(filepath, 'fasta'):
        seq = str(rec.seq).upper()
        if not filter_sequence(seq):
            logging.info(f"  Warning: Skipping sequence with invalid AA: {rec.id}")
            continue
        parts = rec.description.split('|')
        if len(parts) < 2:
            raise ValueError(f"Missing '|' separator: {rec.description}")
        label_str = parts[-1]
        try:
            label = int(label_str)
        except ValueError:
            try:
                label = float(label_str)
            except ValueError:
                raise ValueError(f"Label not int/float: {label_str}")
        seqs.append(seq)
        labels.append(label)
    if any(isinstance(l, float) for l in labels):
        task_type = 'regression'
    else:
        unique = set(labels)
        task_type = 'multiclass' if len(unique) > 2 else 'binary'
    return seqs, labels, task_type

# ============================================================================
# 1. ESM‑2 EMBEDDING EXTRACTION
# ============================================================================

class ESM2EmbeddingExtractor:
    def __init__(self, model_path, base_model_name=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if not ESM_AVAILABLE:
            raise ImportError("Please install: torch, transformers, peft")
        self.device = torch.device(device)
        self.base_model_name = base_model_name or 'facebook/esm2_t6_8M_UR50D'
        self.model, self.tokenizer = self._load_model(model_path)
        self.max_len = self.model.config.max_position_embeddings
        self.hidden_size = self.model.config.hidden_size

    def _load_model(self, model_path):
        tokenizer = EsmTokenizer.from_pretrained(model_path)
        adapter_config = None
        try:
            from peft import PeftConfig
            adapter_config = PeftConfig.from_pretrained(model_path)
        except Exception:
            pass
        if adapter_config is not None:
            logging.info(f"Loading LoRA adapter from {model_path}")
            base_model_name = adapter_config.base_model_name_or_path
            if not os.path.exists(base_model_name):
                base_model_name = getattr(self, 'base_model_name', 'facebook/esm2_t6_8M_UR50D')
                logging.info(f"  Base model path invalid, using {base_model_name} from hub")
            base_model = EsmForMaskedLM.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            logging.info(f"Loading full model from {model_path}")
            model = EsmForMaskedLM.from_pretrained(model_path)
        model = model.to(self.device)
        model.eval()
        return model, tokenizer

    def extract_batch(self, sequences, batch_size=128):
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True,
                                    truncation=True, max_length=self.max_len).to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    mask = inputs.attention_mask.unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings)

# ============================================================================
# 2. DATA LOADING
# ============================================================================

def get_embeddings(seqs, extractor=None, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        logging.info(f"  Loading cached embeddings from {cache_file}")
        data = np.load(cache_file)
        return data['embeddings']
    else:
        if extractor is None:
            raise ValueError("No ESM extractor provided and no cache found")
        logging.info(f"  Extracting embeddings for {len(seqs)} sequences...")
        emb = extractor.extract_batch(seqs)
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez_compressed(cache_file, embeddings=emb)
            logging.info(f"  Saved embeddings to {cache_file}")
        return emb

def check_duplicates_and_contamination(train_seqs, train_labels, test_seqs, test_labels, property_name):
    logging.info(f"\n  Data integrity check for {property_name.upper()}:")
    seen = set()
    uniq_train_seqs, uniq_train_labels = [], []
    for s, l in zip(train_seqs, train_labels):
        if s not in seen:
            seen.add(s)
            uniq_train_seqs.append(s)
            uniq_train_labels.append(l)
    train_set = set(uniq_train_seqs)
    clean_test_seqs, clean_test_labels = [], []
    for s, l in zip(test_seqs, test_labels):
        if s not in train_set:
            clean_test_seqs.append(s)
            clean_test_labels.append(l)
    logging.info(f"    Removed {len(train_seqs)-len(uniq_train_seqs)} train duplicates")
    logging.info(f"    Removed {len(test_seqs)-len(clean_test_seqs)} contaminated test sequences")
    return uniq_train_seqs, uniq_train_labels, clean_test_seqs, clean_test_labels

# ============================================================================
# 3. CLASSIFICATION TRAINING (LR vs DNN, SMOTE optional, SHAP plots)
# ============================================================================


def select_threshold_cv_binary(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = np.argmax(f1)
    best_thresh = thr[best_idx]
    logging.info(f"    CV-selected threshold (max F1): {best_thresh:.4f} (prec={prec[best_idx]:.3f}, rec={rec[best_idx]:.3f})")
    return best_thresh

def evaluate_binary_classifier(model, X, y, threshold=None):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= (threshold if threshold is not None else 0.5)).astype(int)
    return {
        'auc': roc_auc_score(y, proba),
        'ap': average_precision_score(y, proba),
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'mcc': matthews_corrcoef(y, pred)
    }

def train_classification(property_name, train_seqs, train_labels, test_seqs, test_labels,
                         output_dir, extractor, cache_dir, seed, classifier_choice='auto', use_smote=False):

    if test_seqs is not None:
        train_seqs, train_labels, test_seqs, test_labels = check_duplicates_and_contamination(
            train_seqs, train_labels, test_seqs, test_labels, property_name)

    cache_file = os.path.join(cache_dir, f"{property_name}_train_emb.npz") if cache_dir else None
    X_full = get_embeddings(train_seqs, extractor, cache_file)
    y_full = np.array(train_labels)
    logging.info(f"  Training data size: {len(X_full)}")
    is_multiclass = len(set(y_full)) > 2
    if not is_multiclass:
        logging.info(f"Positive class count: {sum(y_full)}")
        if use_smote:
            logging.info("  SMOTE will be applied during training (binary classification).")
    else:
        logging.info(f"Number of classes: {len(set(y_full))}")
        # SMOTE is not applied for multiclass (use class_weight='balanced' instead)
        use_smote = False

    if is_multiclass:
        le = LabelEncoder()
        y_full = le.fit_transform(y_full)
        logging.info(f"    Classes: {le.classes_}")
    else:
        le = None

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    logging.info(f"Using full embedding dimension: {X_full_scaled.shape[1]}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed) if not is_multiclass else KFold(3, shuffle=True, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed,
                            multi_class='multinomial' if is_multiclass else 'auto')
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
    scoring = 'roc_auc_ovr' if is_multiclass else 'roc_auc'
    grid = GridSearchCV(lr, param_grid, cv=3, scoring=scoring, n_jobs=26)
    grid.fit(X_full_scaled, y_full)
    best_C = grid.best_params_['C']
    print(f"Logistic Regression best C = {best_C}")
    lr_best = LogisticRegression(C=best_C, max_iter=1000, class_weight='balanced', random_state=seed,
                                 multi_class='multinomial' if is_multiclass else 'auto')
    models = {'lr': lr_best}

    # DNN (full grid always, n_jobs=32 to avoid threading issues)
    if is_multiclass:
        param_grid_mlp = {
            'hidden_layer_sizes': [(32,), (64,), (128,), (64,32)],
            'alpha': [0.0001, 0.001, 0.01],
        }
        mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam',
                            alpha=0.001, batch_size=32, learning_rate='adaptive',
                            max_iter=100, early_stopping=True, validation_fraction=0.2,
                            random_state=seed)
    else:
        param_grid_mlp = {
            'hidden_layer_sizes': [(32,), (64,), (128,), (64,32)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
        mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam',
                            alpha=0.001, batch_size=32, learning_rate='adaptive',
                            max_iter=100, early_stopping=True, validation_fraction=0.2,
                            random_state=seed)
    logging.info(f"Tuning DNN (full grid, n_jobs=26)...")
    grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=3, scoring=scoring, n_jobs=26)
    grid_mlp.fit(X_full_scaled, y_full)
    best_mlp = grid_mlp.best_estimator_
    logging.info(f"DNN best params: {grid_mlp.best_params_}")
    models['dnn'] = best_mlp

    # Cross-validate to choose best model
    cv_scores = {}
    for name, model in models.items():
        scores = []
        for train_idx, val_idx in cv.split(X_full_scaled, y_full):
            X_tr, X_val = X_full_scaled[train_idx], X_full_scaled[val_idx]
            y_tr, y_val = y_full[train_idx], y_full[val_idx]
            # Apply SMOTE only for binary and if requested (on the training fold)
            if use_smote and not is_multiclass:
                smote = SMOTE(random_state=seed)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            clone_model = clone(model)
            clone_model.fit(X_tr, y_tr)
            if is_multiclass:
                scores.append(accuracy_score(y_val, clone_model.predict(X_val)))
            else:
                scores.append(roc_auc_score(y_val, clone_model.predict_proba(X_val)[:, 1]))
        cv_scores[name] = np.mean(scores)
        print(f"  CV {name}: {cv_scores[name]:.4f}")

    if classifier_choice == 'auto':
        best_name = max(cv_scores, key=cv_scores.get)
        logging.info(f"Auto-selected best model: {best_name} (CV score {cv_scores[best_name]:.4f})")
        final_models = {best_name: models[best_name]}
    elif classifier_choice == 'both':
        final_models = models
        best_name = 'both'
    else:
        best_name = classifier_choice
        final_models = {best_name: models[best_name]}

    # Train selected model(s) on full data (with SMOTE if requested)
    trained_models = {}
    for name, model in final_models.items():
        X_train_full = X_full_scaled
        y_train_full = y_full
        if use_smote and not is_multiclass:
            smote = SMOTE(random_state=seed)
            X_train_full, y_train_full = smote.fit_resample(X_full_scaled, y_full)
            logging.info(f"SMOTE applied to full training set: {len(X_train_full)} samples")
        model.fit(X_train_full, y_train_full)
        trained_models[name] = model

    best_model_obj = list(trained_models.values())[0]

    if not is_multiclass:
        # Compute threshold using CV OOF probabilities (without SMOTE for threshold to keep original distribution)
        oof_proba = np.zeros(len(y_full))
        for train_idx, val_idx in cv.split(X_full_scaled, y_full):
            X_tr, X_val = X_full_scaled[train_idx], X_full_scaled[val_idx]
            y_tr, y_val = y_full[train_idx], y_full[val_idx]
            clone_model = clone(best_model_obj)
            clone_model.fit(X_tr, y_tr)
            oof_proba[val_idx] = clone_model.predict_proba(X_val)[:, 1]
        threshold = select_threshold_cv_binary(y_full, oof_proba)
    else:
        threshold = None

    # Evaluation on test set if available
    if test_seqs is not None and len(test_seqs) > 0:
        cache_test = os.path.join(cache_dir, f"{property_name}_test_emb.npz") if cache_dir else None
        X_test = get_embeddings(test_seqs, extractor, cache_test)
        X_test_scaled = scaler.transform(X_test)
        y_test = np.array(test_labels)
        if is_multiclass and le is not None:
            y_test = le.transform(y_test)

        if is_multiclass:
            test_pred = best_model_obj.predict(X_test_scaled)
            metrics = {
                'accuracy': accuracy_score(y_test, test_pred),
                'f1_macro': f1_score(y_test, test_pred, average='macro'),
                'f1_weighted': f1_score(y_test, test_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
            }
            logging.info(f"\n  Test set performance (multiclass):")
            logging.info(f"    Accuracy: {metrics['accuracy']:.4f}")
            logging.info(f"    F1 macro: {metrics['f1_macro']:.4f}")
            logging.info(f"    F1 weighted: {metrics['f1_weighted']:.4f}")
            logging.info(f"    Confusion matrix:\n{confusion_matrix(y_test, test_pred)}")
            plot_confusion_matrix(confusion_matrix(y_test, test_pred), property_name, output_dir)
        else:
            if best_name == 'both':
                probas = [model.predict_proba(X_test_scaled)[:, 1] for model in trained_models.values()]
                test_proba = np.mean(probas, axis=0)
                logging.info("Using averaged probabilities from LR and DNN")
            else:
                test_proba = best_model_obj.predict_proba(X_test_scaled)[:, 1]
            test_pred = (test_proba >= threshold).astype(int)
            metrics = evaluate_binary_classifier(best_model_obj if best_name != 'both' else None, X_test_scaled, y_test, threshold)
            if best_name == 'both':
                metrics = {
                    'auc': roc_auc_score(y_test, test_proba),
                    'ap': average_precision_score(y_test, test_proba),
                    'accuracy': accuracy_score(y_test, test_pred),
                    'precision': precision_score(y_test, test_pred, zero_division=0),
                    'recall': recall_score(y_test, test_pred, zero_division=0),
                    'f1': f1_score(y_test, test_pred, zero_division=0),
                    'mcc': matthews_corrcoef(y_test, test_pred),
                    'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
                }
            logging.info(f"\n  Test set performance:")
            logging.info(f"    AUC = {metrics['auc']:.4f}, AP = {metrics['ap']:.4f}")
            logging.info(f"    Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}, F1 = {metrics['f1']:.4f}")
            logging.info(f"    Confusion matrix:\n{confusion_matrix(y_test, test_pred)}")
            ci_metrics = bootstrap_metrics(y_test, test_pred, test_proba)
            print("\n  Bootstrapped 95% CIs:")
            for k, v in ci_metrics.items():
                logging.info(f"    {k}: {v}")
            # SHAP for logistic regression (if selected)

            if SHAP_AVAILABLE and best_name == 'lr' and not is_multiclass:
                try:
                    # Define feature names (dimensions 0 to 319)
                    feature_names = [f"dim_{i}" for i in range(X_test_scaled.shape[1])]
                    # Use KernelExplainer for logistic regression
                    # Use a small background sample (100) for speed
                    background = X_full_scaled[np.random.choice(len(X_full_scaled), 100, replace=False)]
                    explainer = shap.KernelExplainer(best_model_obj.predict_proba, background)
                    # Explain test set (first 100)
                    shap_values = explainer.shap_values(X_test_scaled[:100], nsamples=100)
                    # For binary classification, shap_values is a list of two arrays (neg, pos). Use the positive class.
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    save_dict = {
                        'shap_values': shap_values,
                        'X_sample': X_test_scaled[:100],
                        'feature_names': feature_names
                    }
                    with open(os.path.join(output_dir, f"{property_name}_shap_data.pkl"), 'wb') as f:
                        pickle.dump(save_dict, f)
                    print(f"    SHAP data saved for {property_name}")
                except Exception as e:
                    print(f"    SHAP failed: {e}")
            plot_roc_curve(y_test, test_proba, property_name, output_dir)
            plot_pr_curve(y_test, test_proba, property_name, output_dir)
            plot_confusion_matrix(confusion_matrix(y_test, test_pred), property_name, output_dir)
    else:
        # No test set – perform CV and generate OOF plots
        print("\n  No test set – performing 3‑fold CV with full metrics and OOF plots")
        cv_metrics = {}
        if not is_multiclass:
            cv_metrics = {'auc': [], 'ap': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'mcc': []}
        else:
            cv_metrics = {'accuracy': [], 'f1_macro': []}
        oof_proba = np.zeros(len(y_full)) if not is_multiclass else None
        oof_pred = np.zeros(len(y_full))
        for train_idx, val_idx in cv.split(X_full_scaled, y_full):
            X_tr, X_val = X_full_scaled[train_idx], X_full_scaled[val_idx]
            y_tr, y_val = y_full[train_idx], y_full[val_idx]
            if use_smote and not is_multiclass:
                smote = SMOTE(random_state=seed)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            clone_model = clone(best_model_obj)
            clone_model.fit(X_tr, y_tr)
            if is_multiclass:
                pred = clone_model.predict(X_val)
                oof_pred[val_idx] = pred
                cv_metrics['accuracy'].append(accuracy_score(y_val, pred))
                cv_metrics['f1_macro'].append(f1_score(y_val, pred, average='macro'))
            else:
                proba = clone_model.predict_proba(X_val)[:, 1]
                oof_proba[val_idx] = proba
                pred = (proba >= threshold).astype(int)
                oof_pred[val_idx] = pred
                cv_metrics['auc'].append(roc_auc_score(y_val, proba))
                cv_metrics['ap'].append(average_precision_score(y_val, proba))
                cv_metrics['accuracy'].append(accuracy_score(y_val, pred))
                cv_metrics['precision'].append(precision_score(y_val, pred, zero_division=0))
                cv_metrics['recall'].append(recall_score(y_val, pred, zero_division=0))
                cv_metrics['f1'].append(f1_score(y_val, pred, zero_division=0))
                cv_metrics['mcc'].append(matthews_corrcoef(y_val, pred))
        logging.info(f"\n  Cross-validation results (3-fold):")
        for k, v in cv_metrics.items():
            logging.info(f"    {k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")
        if not is_multiclass:
            plot_roc_curve(y_full, oof_proba, property_name, output_dir, suffix="_cv_oof")
            plot_pr_curve(y_full, oof_proba, property_name, output_dir, suffix="_cv_oof")
        plot_confusion_matrix(confusion_matrix(y_full, oof_pred), property_name, output_dir, suffix="_cv_oof")
        metrics = {'cv_metrics': {k: (np.mean(v), np.std(v)) for k, v in cv_metrics.items()}, 'cv_used': True}

    # Save model(s)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{property_name}_model.pkl")
    save_dict = {
        'task_type': 'multiclass' if is_multiclass else 'classification',
        'scaler': scaler,
        'threshold': threshold,
        'label_encoder': le,
        'metrics': metrics,
        'use_embeddings': True,
        'classifier_choice': classifier_choice,
        'used_smote': use_smote,
    }
    if best_name == 'both':
        save_dict['models'] = trained_models
    else:
        save_dict['model'] = best_model_obj
    with open(model_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"  Model saved to {model_path}")
    return metrics

# ============================================================================
# 4. REGRESSION TRAINING (RidgeCV vs MLP, no SMOTE)
# ============================================================================

class CalibratedRidge(BaseEstimator, RegressorMixin):
    """Isotonic calibration wrapper for RidgeCV that is a proper sklearn estimator."""
    def __init__(self, ridge_model=None, isotonic_cal=None):
        self.ridge_model = ridge_model
        self.isotonic_cal = isotonic_cal

    def fit(self, X, y):
        if self.ridge_model is None:
            self.ridge_model_ = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3)
        else:
            self.ridge_model_ = clone(self.ridge_model)
        self.ridge_model_.fit(X, y)
        y_pred_cv = cross_val_predict(self.ridge_model_, X, y, cv=3, method='predict')
        self.isotonic_cal_ = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_cal_.fit(y_pred_cv, y)
        return self

    def predict(self, X):
        raw = self.ridge_model_.predict(X)
        return self.isotonic_cal_.predict(raw)

    def get_params(self, deep=True):
        return {'ridge_model': self.ridge_model, 'isotonic_cal': self.isotonic_cal}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

def train_regression(property_name, train_seqs, train_labels, test_seqs, test_labels,
                     output_dir, extractor, cache_dir, seed=42):
    logging.info(f"{'='*70}")
    logging.info(f"Training REGRESSION model for: {property_name.upper()} (RidgeCV vs MLP)")
    logging.info(f"{'='*70}")

    cache_file = os.path.join(cache_dir, f"{property_name}_train_emb.npz") if cache_dir else None
    X_full = get_embeddings(train_seqs, extractor, cache_file)
    y_full = np.array(train_labels, dtype=float)
    logging.info(f"  Training data size: {len(X_full)}")
    logging.info(f"  Target range: {y_full.min():.3f} – {y_full.max():.3f}")

    # Log transform if skewed
    if np.abs(stats.skew(y_full)) > 1:
        y_transformed = np.log1p(y_full)
        logging.info("  Applied log1p transform to target")
        use_log = True
    else:
        y_transformed = y_full
        use_log = False

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    # Base models (both clonable sklearn estimators)
    base_models = {}

    # 1. RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3)
    base_models['ridge'] = ridge

    # 2. MLPRegressor – full grid always, n_jobs=32
    param_grid_mlp = {
        'hidden_layer_sizes': [(32,), (64,), (128,), (64,32)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    mlp = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', solver='adam',
                       alpha=0.001, batch_size=32, learning_rate='adaptive',
                       max_iter=200, early_stopping=True, validation_fraction=0.2,
                       random_state=seed)
    logging.info(f"  Tuning MLPRegressor (full grid, n_jobs=26)...")
    grid_mlp = GridSearchCV(mlp, param_grid_mlp, cv=3, scoring='neg_mean_squared_error', n_jobs=26)
    grid_mlp.fit(X_full_scaled, y_transformed)
    best_mlp = grid_mlp.best_estimator_
    logging.info(f"    MLP best params: {grid_mlp.best_params_}")
    base_models['mlp'] = best_mlp

    # Cross-validate to choose best model
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    cv_scores = {}
    for name, model in base_models.items():
        scores = []
        for train_idx, val_idx in cv.split(X_full_scaled):
            X_tr, X_val = X_full_scaled[train_idx], X_full_scaled[val_idx]
            y_tr, y_val = y_transformed[train_idx], y_transformed[val_idx]
            clone_model = clone(model)
            clone_model.fit(X_tr, y_tr)
            pred = clone_model.predict(X_val)
            scores.append(spearmanr(y_val, pred)[0])
        cv_scores[name] = np.mean(scores)
        logging.info(f"  CV {name}: Spearman ρ = {cv_scores[name]:.4f}")

    best_name = max(cv_scores, key=cv_scores.get)
    logging.info(f"  Auto-selected best model: {best_name} (CV ρ = {cv_scores[best_name]:.4f})")

    # Train final model
    if best_name == 'ridge':
        final_model = CalibratedRidge(ridge_model=base_models['ridge'])
        final_model.fit(X_full_scaled, y_transformed)
    else:
        final_model = clone(base_models['mlp'])
        final_model.fit(X_full_scaled, y_transformed)

    # Evaluation on test set if available
    if test_seqs is not None and len(test_seqs) > 0:
        cache_test = os.path.join(cache_dir, f"{property_name}_test_emb.npz") if cache_dir else None
        X_test = get_embeddings(test_seqs, extractor, cache_test)
        X_test_scaled = scaler.transform(X_test)
        y_test = np.array(test_labels, dtype=float)
        pred_log = final_model.predict(X_test_scaled)
        pred = np.expm1(pred_log) if use_log else pred_log
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        pearson_r, p_pearson = pearsonr(y_test, pred)
        spearman_r, p_spearman = spearmanr(y_test, pred)
        metrics = {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'pearson': pearson_r, 'spearman': spearman_r,
            'p_pearson': p_pearson, 'p_spearman': p_spearman
        }
        logging.info(f"\n  Test set performance:")
        logging.info(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        logging.info(f"    Pearson r: {pearson_r:.4f} (p={p_pearson:.2e})")
        logging.info(f"    Spearman ρ: {spearman_r:.4f} (p={p_spearman:.2e})")
        plot_regression_scatter(y_test, pred, property_name, output_dir)
        plot_regression_residuals(y_test, pred, property_name, output_dir)
    else:
        logging.info("\n  No test set – performing 3‑fold CV with OOF plots")
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        cv_rmse, cv_mae, cv_r2, cv_rho = [], [], [], []
        oof_pred = np.zeros(len(y_full))
        for train_idx, val_idx in cv.split(X_full_scaled):
            X_tr, X_val = X_full_scaled[train_idx], X_full_scaled[val_idx]
            y_tr, y_val = y_transformed[train_idx], y_transformed[val_idx]
            clone_model = clone(final_model)
            clone_model.fit(X_tr, y_tr)
            pred_log = clone_model.predict(X_val)
            pred_cv = np.expm1(pred_log) if use_log else pred_log
            oof_pred[val_idx] = pred_cv
            y_val_orig = np.expm1(y_val) if use_log else y_val
            cv_rmse.append(np.sqrt(mean_squared_error(y_val_orig, pred_cv)))
            cv_mae.append(mean_absolute_error(y_val_orig, pred_cv))
            cv_r2.append(r2_score(y_val_orig, pred_cv))
            cv_rho.append(spearmanr(y_val_orig, pred_cv)[0])
        logging.info(f"    CV RMSE:       {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
        logging.info(f"    CV MAE:        {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}")
        logging.info(f"    CV R²:         {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
        logging.info(f"    CV Spearman ρ: {np.mean(cv_rho):.4f} ± {np.std(cv_rho):.4f}")
        y_orig = np.expm1(y_transformed) if use_log else y_transformed
        plot_regression_scatter(y_orig, oof_pred, property_name, output_dir, suffix="_cv_oof")
        plot_regression_residuals(y_orig, oof_pred, property_name, output_dir, suffix="_cv_oof")
        metrics = {'rmse': (np.mean(cv_rmse), np.std(cv_rmse)), 'cv_used': True}

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{property_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'task_type': 'regression',
            'model': final_model,
            'scaler': scaler,
            'use_log_transform': use_log,
            'best_regressor': best_name,
            'metrics': metrics,
            'use_embeddings': True
        }, f)
    logging.info(f"  Model saved to {model_path}")
    return metrics

# ============================================================================
# 5. HELPER FUNCTIONS (bootstrap, plots)
# ============================================================================

# Plotting functions (all with plt.close('all'))
def plot_roc_curve(y_true, y_prob, prop_name, out_dir, suffix=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc_val:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC – {prop_name.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prop_name}_roc{suffix}.png"), dpi=120)
    plt.savefig(os.path.join(out_dir, f"{prop_name}_roc{suffix}.svg"))
    plt.close('all')

def plot_pr_curve(y_true, y_prob, prop_name, out_dir, suffix=""):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec, lw=2, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision‑Recall – {prop_name.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prop_name}_pr{suffix}.png"), dpi=120)
    plt.savefig(os.path.join(out_dir, f"{prop_name}_pr{suffix}.svg"))
    plt.close('all')
    def __init__(self, model_path, base_model_name=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if not ESM_AVAILABLE:
            raise ImportError("Please install: torch, transformers, peft")
        self.device = torch.device(device)
        self.base_model_name = base_model_name or 'facebook/esm2_t6_8M_UR50D'
        self.model, self.tokenizer = self._load_model(model_path)
        self.max_len = self.model.config.max_position_embeddings
        self.hidden_size = self.model.config.hidden_size
def plot_confusion_matrix(cm, prop_name, out_dir, suffix=""):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix – {prop_name.capitalize()}')
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prop_name}_cm{suffix}.png"), dpi=120)
    plt.savefig(os.path.join(out_dir, f"{prop_name}_cm{suffix}.svg"))
    plt.close('all')

def plot_regression_scatter(y_true, y_pred, prop_name, out_dir, suffix=""):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('True value')
    plt.ylabel('Predicted')
    plt.title(f'{prop_name.capitalize()} – Predicted vs True')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prop_name}_scatter{suffix}.png"), dpi=120)
    plt.savefig(os.path.join(out_dir, f"{prop_name}_scatter{suffix}.svg"))
    plt.close('all')

def plot_regression_residuals(y_true, y_pred, prop_name, out_dir, suffix=""):
    residuals = y_true - y_pred
    plt.figure(figsize=(5,4))
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'{prop_name.capitalize()} – Residual histogram')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prop_name}_residual_hist{suffix}.png"), dpi=120)
    plt.savefig(os.path.join(out_dir, f"{prop_name}_residual_hist{suffix}.svg"))
    plt.close('all')
    plt.figure(figsize=(5,5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{prop_name.capitalize()} – QQ plot of residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prop_name}_qq{suffix}.png"), dpi=120)
    plt.savefig(os.path.join(out_dir, f"{prop_name}_qq{suffix}.svg"))
    plt.close('all')

# ============================================================================
# 6. PREDICTION MODE (unchanged)
# ============================================================================

def load_models(model_dir):
    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith('_model.pkl'):
            prop = fname.replace('_model.pkl', '')
            with open(os.path.join(model_dir, fname), 'rb') as f:
                models[prop] = pickle.load(f)
    return models

def predict_fasta(fasta_path, model_dir, output_csv, extractor=None, cache_dir=None, classifier_choice='auto'):
    logging.info(f"\nLoading models from {model_dir}...")
    models = load_models(model_dir)
    if not models:
        logging.info("ERROR: No trained models found.")
        sys.exit(1)

    seqs = []
    ids = []
    for rec in SeqIO.parse(fasta_path, 'fasta'):
        seq = str(rec.seq).upper()
        if not filter_sequence(seq):
            logging.info(f"  Warning: Skipping sequence with invalid AA: {rec.id}")
            continue
        seqs.append(seq)
        ids.append(rec.id)

    if len(seqs) == 0:
        logging.info("ERROR: No valid sequences after filtering.")
        sys.exit(1)

    logging.info(f"Extracting embeddings for {len(seqs)} sequences...")
    if extractor is None:
        raise ValueError("ESM‑2 extractor required for prediction")
    X = extractor.extract_batch(seqs)

    results = {'ID': ids, 'Sequence': seqs}
    for prop, data in models.items():
        scaler = data['scaler']
        X_scaled = scaler.transform(X)
        task = data['task_type']
        if task == 'classification':
            if 'models' in data:
                lr_model = data['models']['lr']
                dnn_model = data['models']['dnn']
                proba_lr = lr_model.predict_proba(X_scaled)[:, 1]
                proba_dnn = dnn_model.predict_proba(X_scaled)[:, 1]
                if classifier_choice == 'lr':
                    proba = proba_lr
                elif classifier_choice == 'dnn':
                    proba = proba_dnn
                else:
                    proba = (proba_lr + proba_dnn) / 2
            else:
                model = data['model']
                proba = model.predict_proba(X_scaled)[:, 1]
            threshold = data['threshold']
            pred = (proba >= threshold).astype(int)
            results[f"{prop}_prob"] = proba
            results[f"{prop}_pred"] = pred
        elif task == 'multiclass':
            model = data['model']
            pred = model.predict(X_scaled)
            le = data.get('label_encoder')
            if le is not None:
                pred = le.inverse_transform(pred)
            results[f"{prop}_pred"] = pred
        else:  # regression
            model = data['model']
            pred_log = model.predict(X_scaled)
            if data.get('use_log_transform', False):
                pred = np.expm1(pred_log)
            else:
                pred = pred_log
            results[f"{prop}_pred"] = pred
        logging.info(f"  {prop}: predictions done")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    logging.info(f"\nPredictions saved to {output_csv}")
    logging.info(df.head())

# ============================================================================
# 7. MAIN (with --use_smote flag)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi‑Property Predictor using ESM‑2 embeddings (LR vs DNN, SMOTE optional)')
    parser.add_argument('--embedding_model_path', help='Path to ESM‑2 model or LoRA checkpoint')
    parser.add_argument('--embedding_cache_dir', default='./embedding_cache', help='Directory to cache embeddings (.npz)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--base_model_name', default='facebook/esm2_t6_8M_UR50D',
                        help='Base ESM‑2 model name or path')
    parser.add_argument('--classifier', choices=['auto', 'lr', 'dnn', 'both'], default='auto',
                        help='Which classifier to use: auto (CV selects best), lr, dnn, both (average for prediction)')
    parser.add_argument('--use_smote', action='store_true', help='Apply SMOTE for binary classification (oversampling minority class)')
    # Property arguments
    parser.add_argument('--amp_train', help='Antimicrobial train FASTA')
    parser.add_argument('--amp_test', help='Antimicrobial test FASTA')
    parser.add_argument('--fouling_train', help='Fouling train FASTA')
    parser.add_argument('--fouling_test', help='Fouling test FASTA')
    parser.add_argument('--hemo_train', help='Hemolytic train FASTA')
    parser.add_argument('--hemo_test', help='Hemolytic test FASTA')
    parser.add_argument('--sol_train', help='Solubility train FASTA')
    parser.add_argument('--sol_test', help='Solubility test FASTA')
    parser.add_argument('--tox_train', help='Toxicity train FASTA')
    parser.add_argument('--tox_test', help='Toxicity test FASTA')
    parser.add_argument('--allergen_train', help='Allergen train FASTA')
    parser.add_argument('--allergen_test', help='Allergen test FASTA')
    parser.add_argument('--stability_train', help='Stability train FASTA (binary)')
    parser.add_argument('--stability_test', help='Stability test FASTA')
    parser.add_argument('--qsi_train', help='Quorum sensing inhibition train FASTA')
    parser.add_argument('--aox_train', help='AOX train FASTA')
    parser.add_argument('--bitter_train', help='Bitterness train FASTA (regression)')
    parser.add_argument('--bitter_test', help='Bitterness test FASTA')
    parser.add_argument('--half_train', help='Half-life train FASTA (regression)')
    parser.add_argument('--half_test', help='Half-life test FASTA')
    # Other
    parser.add_argument('--predict', help='FASTA file for prediction')
    parser.add_argument('--model_dir', default='./models_esm2', help='Directory to save/load models')
    parser.add_argument('--output_csv', default='predictions.csv', help='Output CSV for predictions')
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.embedding_cache_dir, exist_ok=True)

    extractor = None
    if args.embedding_model_path:
        if not ESM_AVAILABLE:
            logging.info("ERROR: transformers/peft/torch not installed. Please install: pip install torch transformers peft")
            sys.exit(1)
        extractor = ESM2EmbeddingExtractor(args.embedding_model_path, args.base_model_name, args.device)

    if args.predict:
        if extractor is None:
            logging.info("ERROR: --embedding_model_path required for prediction")
            sys.exit(1)
        predict_fasta(args.predict, args.model_dir, args.output_csv, extractor, args.embedding_cache_dir, args.classifier)
        return

    # Training mode – all properties are binary now (including stability)
    binary_tasks = [
        ('antimicrobial', args.amp_train, args.amp_test),
        ('fouling', args.fouling_train, args.fouling_test),
        ('hemolytic', args.hemo_train, args.hemo_test),
        ('solubility', args.sol_train, args.sol_test),
        ('toxicity', args.tox_train, args.tox_test),
        ('allergen', args.allergen_train, args.allergen_test),
        ('stability', args.stability_train, args.stability_test),
        ('qsi', args.qsi_train, None),
        ('aox', args.aox_train, None),
        ('bitterness', args.bitter_train, args.bitter_test),
        ('halflife', args.half_train, None),
    ]
    regression_tasks = []

    all_metrics = {}

    for prop, train_path, test_path in binary_tasks:
        if train_path:
            logging.info(f"\nLoading {prop} binary data...")
            train_seqs, train_labels, _ = load_fasta_labels(train_path)
            test_seqs, test_labels = (load_fasta_labels(test_path)[:2] if test_path else (None, None))
            metrics = train_classification(prop, train_seqs, train_labels, test_seqs, test_labels,
                                           args.model_dir, extractor, args.embedding_cache_dir,
                                           classifier_choice=args.classifier, use_smote=args.use_smote)
            all_metrics[prop] = metrics

    for prop, train_path, test_path in regression_tasks:
        if train_path:
            logging.info(f"\nLoading {prop} regression data...")
            train_seqs, train_labels, _ = load_fasta_labels(train_path)
            test_seqs, test_labels = (load_fasta_labels(test_path)[:2] if test_path else (None, None))
            metrics = train_regression(prop, train_seqs, train_labels, test_seqs, test_labels,
                                       args.model_dir, extractor, args.embedding_cache_dir)
            all_metrics[prop] = metrics

    # Summary report
    

if __name__ == '__main__':
    main()
