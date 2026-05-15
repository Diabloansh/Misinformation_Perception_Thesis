import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from transformers import AutoTokenizer
CLEAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.join(CLEAN_ROOT, 'main_route_classifier')
sys.path.insert(0, os.path.join(MAIN_DIR, 'src'))
from dataset import create_datasets
from model import build_model

def load_trained_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    class_weights = state_dict.get('class_weights')
    model = build_model(config, class_weights=class_weights).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    tokenizer = checkpoint.get('tokenizer') or AutoTokenizer.from_pretrained(config.model_name)
    return (model, config, tokenizer)

def main():
    parser = argparse.ArgumentParser(description='Generate Test1 plots for the main route classifier')
    parser.add_argument('--model', default=os.path.join(MAIN_DIR, 'results_main', 'best_model.pt'))
    parser.add_argument('--data', default=os.path.join(MAIN_DIR, 'Dataset', 'Testing_Dataset', 'Test1.json'))
    parser.add_argument('--output', default=os.path.join(MAIN_DIR, 'test1_plots'))
    args = parser.parse_args()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    model, config, _ = load_trained_model(args.model, device)
    _, _, test_dataset, _ = create_datasets(args.data, config, test_filepath=args.data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    y_true, y_score = ([], [])
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            probs = torch.softmax(outputs['logits'], dim=-1)
            y_true.extend(batch['labels'].numpy())
            y_score.extend(probs.cpu().numpy())
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = y_score.argmax(axis=-1)
    class_names = config.label_names
    n_classes = config.num_labels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    y_bin = np.zeros((y_true.size, n_classes))
    y_bin[np.arange(y_true.size), y_true] = 1
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Test1 Confusion Matrix')
    plt.savefig(os.path.join(args.output, 'fig_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_score[:, idx])
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr), estimator_name=class_names[idx]).plot(ax=ax, color=color)
    ax.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)')
    ax.set_title('Test1 ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(args.output, 'fig_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    auprc_scores = []
    for idx, color in zip(range(n_classes), colors):
        PrecisionRecallDisplay.from_predictions(y_bin[:, idx], y_score[:, idx], name=class_names[idx], color=color, ax=ax)
        auprc_scores.append(average_precision_score(y_bin[:, idx], y_score[:, idx]))
    ax.set_title(f'Test1 Precision-Recall Curve (Macro AUPRC: {np.mean(auprc_scores):.4f})')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(args.output, 'fig_auprc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Test1 plots saved to {args.output}')
if __name__ == '__main__':
    main()
