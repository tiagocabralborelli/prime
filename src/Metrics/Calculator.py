from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
    confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)

import pandas as pd

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
