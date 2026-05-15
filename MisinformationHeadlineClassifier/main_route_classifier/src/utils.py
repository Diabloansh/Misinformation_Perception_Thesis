import json
import logging
import random
from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(log_level: str='INFO'):
    logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(probabilities: np.ndarray, true_labels: np.ndarray, label_names: List[str]) -> Dict:
    # pick the class with the highest softmax probability
    pred_labels = probabilities.argmax(axis=1)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average=None, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    try:
        roc_auc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
    except ValueError:
        # this can happen if a split is missing a class
        roc_auc = 0.0
    cm = confusion_matrix(true_labels, pred_labels).tolist()
    metrics: Dict = {'accuracy': float(accuracy), 'macro_f1': float(macro_f1), 'weighted_f1': float(weighted_f1), 'macro_precision': float(macro_precision), 'macro_recall': float(macro_recall), 'roc_auc': float(roc_auc), 'confusion_matrix': cm}
    for i, name in enumerate(label_names):
        metrics[f'{name}_precision'] = float(precision[i])
        metrics[f'{name}_recall'] = float(recall[i])
        metrics[f'{name}_f1'] = float(f1[i])
        metrics[f'{name}_support'] = int(support[i])
    return metrics

def print_classification_report(probabilities: np.ndarray, true_labels: np.ndarray, label_names: List[str]):
    pred_labels = probabilities.argmax(axis=1)
    print(classification_report(true_labels, pred_labels, target_names=label_names, digits=4))

def save_metrics(metrics: Dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)
