from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
    confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import numpy as np
import pandas as pd
import os

class Performance:
    def __init__(self, true, pred):
        self.true = true
        self.pred = pred

    def measure(self):
        self.metrics = {
            'auc': roc_auc_score(self.true, self.pred),
            'ap': average_precision_score(self.true, self.pred),
            'accuracy': accuracy_score(self.true, self.pred),
            'precision': precision_score(self.true, self.pred, zero_division=0),
            'recall': recall_score(self.true, self.pred, zero_division=0),
            'f1': f1_score(self.true, self.pred, zero_division=0),
            'mcc': matthews_corrcoef(self.true, self.pred),
            'confusion_matrix': confusion_matrix(self.true, self.pred).tolist()
                }

# class SHAP:


def bootstrap_metrics(y_true, y_pred, y_prob=None, n_bootstrap=1000, ci=95):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = None if y_prob is None else np.asarray(y_prob)
    rng = np.random.default_rng(42)
    n = len(y_true)
    indices = np.arange(n)
    auc_list, ap_list, acc_list, prec_list, rec_list, f1_list, mcc_list = [], [], [], [], [], [], []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(indices, size=n, replace=True)
        y_true_boot = y_true[boot_idx]
        y_pred_boot = y_pred[boot_idx]
        acc_list.append(accuracy_score(y_true_boot, y_pred_boot))
        prec_list.append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        rec_list.append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        f1_list.append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        mcc_list.append(matthews_corrcoef(y_true_boot, y_pred_boot))
        if y_prob is not None and np.unique(y_true_boot).size == 2:
            y_prob_boot = y_prob[boot_idx]
            auc_list.append(roc_auc_score(y_true_boot, y_prob_boot))
            ap_list.append(average_precision_score(y_true_boot, y_prob_boot))
    def summarize(vals):
        low = np.percentile(vals, (100-ci)/2)
        high = np.percentile(vals, 100 - (100-ci)/2)
        return {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)),
            'ci_low': float(low),
            'ci_high': float(high),
        }
    res = {}
    if auc_list:
        res['auc'] = summarize(auc_list)
        res['ap'] = summarize(ap_list)
    res['accuracy'] = summarize(acc_list)
    res['precision'] = summarize(prec_list)
    res['recall'] = summarize(rec_list)
    res['f1'] = summarize(f1_list)
    res['mcc'] = summarize(mcc_list)
    return res


def plot_binary_evaluation(y_true, y_pred, y_prob, output_dir, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {auc_value:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {name}')
    plt.legend()
    plt.tight_layout()
    for extension in ('png', 'svg'):
        plt.savefig(os.path.join(output_dir, f'{name}_roc.{extension}'))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_value = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f'AP = {ap_value:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall - {name}')
    plt.legend()
    plt.tight_layout()
    for extension in ('png', 'svg'):
        plt.savefig(os.path.join(output_dir, f'{name}_pr.{extension}'))
    plt.close()

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    ticks = np.arange(matrix.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    midpoint = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(column, row, str(matrix[row, column]), ha='center', va='center',
                     color='white' if matrix[row, column] > midpoint else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    for extension in ('png', 'svg'):
        plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.{extension}'))
    plt.close()