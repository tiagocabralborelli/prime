import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_val_predict
from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
    confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.base import clone


def select_best_classifier(cv_models, cv_scores, mode):
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

class Coach:
    def __init__(self, train_emb, train_labels, seed):
        self.train_emb = train_emb
        self.train_labels = train_labels
        self.seed = seed

    def preprocess(self, is_multiclass = False):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._isMulticlass = is_multiclass
        if is_multiclass:            
            self.train_labels = self.label_encoder .fit_transform(self.train_labels)
            
            
        else:
            self.label_encoder = None
            
        self.train_emb = self.scaler.fit_transform(self.train_emb)

    def select_binary_classifier(self,model_dict, classifier_choice = False, use_smote=False):

        if self._isMulticlass:
            scoring = "roc_auc_ovr"
            cv = KFold(3, shuffle=True, random_state=42)
        else:
            scoring = "roc_auc"
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)


        print("Model options for CV")
        print(model_dict)
        models = {}
        lr_model = model_dict["linear"]["lr"]
        lr_model.random_state = self.seed
        grid = GridSearchCV(
            estimator =  lr_model,
            param_grid = model_dict["linear"]["param_grid"],
            scoring = scoring,
            n_jobs = model_dict["linear"]["n_jobs"],
            cv = 3
            )
        best_c = grid.fit(
            self.train_emb,
            self.train_labels
                )
        models["LR"] = grid.best_estimator_

        del grid, best_c

        mlp_model = model_dict["non-linear"]['mlp']
        mlp_model.random_state = self.seed
        grid_mlp = GridSearchCV(
                    estimator=mlp_model,
                    param_grid= model_dict["non-linear"]["param_grid"]["multiclass"] if self._isMulticlass == True else model_dict["non-linear"]["param_grid"]["binary"],
                    scoring = scoring,
                    n_jobs = model_dict["non-linear"]["n_jobs"],
                    cv = 3)

        best_c = grid_mlp.fit(
            self.train_emb,
            self.train_labels
                )
        
        models["MLP"] = grid_mlp.best_estimator_

        print("Best settings after CV")
        print(models)

        del grid_mlp, best_c

        self.cross_val_scores = {}
        for name, model in models.items():
            scores = []    
            for train_idx, val_idx in cv.split(self.train_emb,self.train_labels):
                X_tr, X_val = self.train_emb[train_idx], self.train_emb[val_idx]
                y_tr, y_val = self.train_labels[train_idx], self.train_labels[val_idx]
                if use_smote:
                    fold_smote = SMOTE(random_state=self.seed)
                    X_tr, y_tr = fold_smote.fit_resample(X_tr, y_tr)
                # print(model)
                clone_model = clone(model)
                clone_model.fit(X_tr, y_tr)
                if self._isMulticlass:
                    scores.append(accuracy_score(y_val, clone_model.predict(X_val)))
                else:
                    scores.append(roc_auc_score(y_val, clone_model.predict_proba(X_val)[:, 1]))
                    
            self.cross_val_scores[name] = np.mean(scores)

        print("CV resulting scores")
        print(self.cross_val_scores)

        if classifier_choice:
            best_model = select_best_classifier(cv_models = models, cv_scores = self.cross_val_scores, mode = classifier_choice)
            print("Selected classifier")
            print(best_model)
           
        elif not classifier_choice:
            raise "Selection Mode Not Found"
                

        #Training selected classifier
        self.trained_best_models = {}
        for name, model in best_model.items():
            if use_smote:
                final_smote = SMOTE(random_state=self.seed)
                self.train_emb, self.train_labels = final_smote.fit_resample(self.train_emb, self.train_labels)
            model.fit(self.train_emb,self.train_labels)
            self.trained_best_models[name] = model

        self.best_model_obj = list(self.trained_best_models.values())[0]


        # Compute threshold using CV OOF probabilities (without SMOTE for threshold to keep original distribution)
        oof_proba = np.zeros(len(self.train_labels))
        for train_idx, val_idx in cv.split(self.train_emb, self.train_labels):
            X_tr, X_val = self.train_emb[train_idx], self.train_emb[val_idx]
            y_tr, y_val = self.train_labels[train_idx], self.train_labels[val_idx]
            clone_model = clone(self.best_model_obj)
            clone_model.fit(X_tr, y_tr)
            oof_proba[val_idx] = clone_model.predict_proba(X_val)[:, 1]
        self.threshold = select_threshold_cv_binary(self.train_labels, oof_proba)


        del self.train_emb, self.train_labels

    def predict(self, test_emb, test_labels):
        # self.test_emb = test_emb
        # self.test_labels = test_labels
        test_proba = self.best_model_obj.predict_proba(test_emb)[:, 1]
        self.test_pred = (test_proba >= self.threshold).astype(int)