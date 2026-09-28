import click
import shutil
import logging
import torch
import os
import sys
from Utils import *

try:
    import torch
    from transformers import EsmTokenizer, EsmForMaskedLM
    from peft import PeftModel
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False


@click.group()
def main():
    pass

@main.command(name = "train")
@click.option('--embedding_model_path', help='Path to ESM‑2 model or LoRA checkpoint')
@click.option('--embedding_cache_dir',  default='./embedding_cache', help='Directory to cache embeddings (.npz)')
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
@click.option('--classifier', type = click.Choice(['auto', 'lr', 'dnn', 'both']), default='auto', help = 'Which classifier to use: auto (CV selects best), lr, dnn, both (average for prediction)')
@click.option('--base_model_name', default='facebook/esm2_t6_8M_UR50D',help='Base ESM‑2 model name or path')
@click.option('--use_smote', is_flag=True,  help='Apply SMOTE for binary classification (oversampling minority class)')

@click.option('--amp_train',    help='Antimicrobial train FASTA')
@click.option('--amp_test',     help='Antimicrobial test FASTA')
@click.option('--fouling_train', help='Fouling train FASTA')
@click.option('--fouling_test' , help='Fouling test FASTA')
@click.option('--hemo_train',  help='Hemolytic train FASTA')
@click.option('--hemo_test', help='Hemolytic test FASTA')
@click.option('--sol_train', help='Solubility train FASTA')
@click.option('--sol_test', help='Solubility test FASTA')
@click.option('--tox_train', help='Toxicity train FASTA')
@click.option('--tox_test', help='Toxicity test FASTA')
# @click.option('--bitter_train', help='Bitterness train FASTA')
# @click.option('--bitter_test', help='Bitterness test FASTA')


@click.option('--log_file',  help = "File name for saving log information", default = 'esmp.log')
@click.option('--model_dir', default='./models_esm2', help='Directory to save/load models')
def train(
    model_dir, embedding_cache_dir, embedding_model_path, base_model_name, device, classifier, use_smote, log_file,
    amp_train, amp_test,
    fouling_train,fouling_test,
    hemo_train,hemo_test,
    sol_train,sol_test,
    tox_train,tox_test,
    # bitter_train, bitter_test
    ):

    #Settings and fixed params
    SEED = 42
    ConfigureLogging(log_file)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(embedding_cache_dir, exist_ok=True)

    extractor = None
    if embedding_model_path:
        if not ESM_AVAILABLE:
            print("ERROR: transformers/peft/torch not installed. Please install: pip install torch transformers peft")
            sys.exit(1)
        extractor = ESM2EmbeddingExtractor(embedding_model_path, base_model_name, device)

    binary_tasks = [
        ('antimicrobial', amp_train, amp_test),
        ('fouling', fouling_train, fouling_test),
        ('hemolytic', hemo_train, hemo_test),
        ('solubility', sol_train, sol_test),
        ('toxicity', tox_train, tox_test),
        # ('bitterness', bitter_train, bitter_test)
    ]
    #========================================================
    ## CLASSIFIERS
    #========================================================
    logging.info(f"{'='*70}")
    logging.info(f"Training Classifier (LR vs DNN)")
    logging.info(f"{'='*70}")
    all_metrics = {}
    for prop, train_path, test_path in binary_tasks:
        if train_path:
            logging.info(f"\nLoading {prop} binary data...")
            train_seqs, train_labels, _ = load_fasta_labels(train_path)
            test_seqs, test_labels = (load_fasta_labels(test_path)[:2] if test_path else (None, None))
            metrics = train_classification(prop, train_seqs, train_labels, test_seqs, test_labels,
                model_dir, extractor, embedding_cache_dir, classifier_choice=classifier, use_smote=use_smote, seed = SEED)
            all_metrics[prop] = metrics
    #========================================================
    ## REGRESSION - Not being used
    #========================================================
    logging.info(f"{'='*70}")
    logging.info(f"Training Regression (LR vs DNN)")
    logging.info(f"{'='*70}")
    all_metrics = {}
    for prop, train_path, test_path in regression_tasks:
        if train_path:
            logging.info(f"\nLoading {prop} regression data...")
            train_seqs, train_labels, _ = load_fasta_labels(train_path)
            test_seqs, test_labels = (load_fasta_labels(test_path)[:2] if test_path else (None, None))
            metrics = train_regression(prop, train_seqs, train_labels, test_seqs, test_labels,
                                    model_dir, extractor,embedding_cache_dir, seed = SEED)
            all_metrics[prop] = metrics
            

    regression_tasks = []
    all_metrics = {}
    for prop, train_path, test_path in binary_tasks:
        if train_path:
            logging.info(f"\nLoading {prop} binary data...")
            train_seqs, train_labels, _ = load_fasta_labels(train_path)
            test_seqs, test_labels = (load_fasta_labels(test_path)[:2] if test_path else (None, None))
            metrics = train_classification(prop, train_seqs, train_labels, test_seqs, test_labels,
                                                model_dir, extractor, embedding_cache_dir,
                                                classifier_choice = classifier, use_smote=use_smote, seed= SEED)
            all_metrics[prop] = metrics

    for prop, train_path, test_path in regression_tasks:
        if train_path:
            logging.info(f"\nLoading {prop} regression data...")
            train_seqs, train_labels, _ = load_fasta_labels(train_path)
            test_seqs, test_labels = (load_fasta_labels(test_path)[:2] if test_path else (None, None))
            metrics = train_regression(prop, train_seqs, train_labels, test_seqs, test_labels,
                                            model_dir, extractor,embedding_cache_dir)
            all_metrics[prop] = metrics    
    logging.info("ALL METRICS HERE")
    logging.info(all_metrics)
    report_path = os.path.join(model_dir, 'training_summary.txt')

    with open(report_path, 'w') as f:
        f.write(f"ESM‑2 Multi‑Property Training Summary (LR vs DNN, SMOTE={use_smote})\nDate: {datetime.now()}\n\n")
        for prop, m in all_metrics.items():
            f.write(f"\n{prop.upper()}\n")
            if m.get('cv_used', False):
                if 'cv_metrics' in m:
                    for k, (mean_val, std_val) in m['cv_metrics'].items():
                        f.write(f"  CV {k.upper()}: {mean_val:.4f} ± {std_val:.4f}\n")
                else:
                    for k in ['auc', 'accuracy', 'rmse']:
                        if k in m:
                            mean_val, std_val = m[k]
                            f.write(f"  CV {k.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
            else:
                for k, v in m.items():
                    if k in ['confusion_matrix', 'p_pearson', 'p_spearman', 'cv_used', 'cv_metrics']:
                        continue
                    if isinstance(v, (int, float)):
                        f.write(f"  {k.capitalize()}: {v:.4f}\n")
                if 'confusion_matrix' in m:
                    f.write(f"  Confusion matrix: {m['confusion_matrix']}\n")
    logging.info(f"\nSummary report saved to {report_path}")


if __name__ == '__main__':
    main()