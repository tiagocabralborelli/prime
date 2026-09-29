import click 
from Loader.Dataset import DataSet
from Embedder.Extractor import EmbedderESM, EmbedderLoRAESM
from Coach.Classifiers import Coach
from Metrics.Calculator import Performance
import os
import pickle

from transformers import EsmTokenizer, EsmForMaskedLM
from sklearn.metrics import precision_recall_curve
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


@click.command()
@click.option('--train', required = True, help = "Training FASTA file with sequence labels in the FASTA IDs")
@click.option('--test', required = True,  help = "Test FASTA file with sequence labels in the FASTA IDs")
@click.option('--name', required = True,  help = "Project name")
@click.option('--base-model', required = True, help = "Pre-trained model folder")
@click.option('--seed', required = False, help = "Random state")
@click.option('--smote', is_flag=True, default=False, help="Enable SMOTE")

def main(train, test, name, base_model, seed=42, smote=False):
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


    # print("Try LoRA")
    # ExtratorESM_LoRA = EmbedderLoRAESM(
    #     path = base_model,
    #     device = "cuda"
    # )
    # ExtratorESM_LoRA.loadModel()

    #Encode sequences into embedding matrices
    ExtratorESM = EmbedderESM(
        path = base_model,
        device = "cuda"
        )
    ExtratorESM.loadModel(tokenizer=EsmTokenizer, model_loader = EsmForMaskedLM)


    #Train classifiers
    train_embedded = ExtratorESM.embedding(DataObj.train_seq)
    train_lables = DataObj.train_id
    ModelSelectorObj = Coach(
        train_emb = train_embedded,
        train_labels = train_lables,
        seed = seed)
    ModelSelectorObj.preprocess()
    ModelSelectorObj.select_binary_classifier(model_dict=HEADS, classifier_choice='auto', use_smote = smote)

    ModelSelectorObj.predict(
        test_emb=ExtratorESM.embedding(DataObj.test_seq),
        test_labels= DataObj.test_id)

    PerformanceObj = Performance(
        true = DataObj.test_id,
        pred = ModelSelectorObj.test_pred
    )
    PerformanceObj.measure()
    print(PerformanceObj.metrics)



    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    artifact = {
    "model": ModelSelectorObj.best_model_obj,
    "threshold": ModelSelectorObj.threshold,
    "scaler": ModelSelectorObj.scaler,
    "seed": seed,
    "name": name,
}

    with open(os.path.join(save_dir, f"{name}_model.pkl"), "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved model to {os.path.join(save_dir, f'{name}_model.pkl')}")
if __name__ == "__main__":
    main()