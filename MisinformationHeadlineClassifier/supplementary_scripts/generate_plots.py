import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
CLEAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.join(CLEAN_ROOT, 'main_route_classifier')
sys.path.insert(0, os.path.join(MAIN_DIR, 'src'))
from dataset import create_datasets
from model import build_model
from utils import set_seed

def plot_learning_curves(log_path: str, output_dir: str):
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    epochs = [ep['epoch'] for ep in log_data['epochs']]
    train_loss = [ep['train_loss'] for ep in log_data['epochs']]
    val_loss = [ep.get('val_loss', ep['val_metrics']['loss']) for ep in log_data['epochs']]
    train_f1 = [ep['train_metrics']['macro_f1'] for ep in log_data['epochs']]
    val_f1 = [ep['val_metrics']['macro_f1'] for ep in log_data['epochs']]
    train_acc = [ep['train_metrics']['accuracy'] for ep in log_data['epochs']]
    val_acc = [ep['val_metrics']['accuracy'] for ep in log_data['epochs']]
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fig_01_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_f1, 'bo-', label='Train Macro F1')
    plt.plot(epochs, val_f1, 'ro-', label='Val Macro F1')
    plt.plot(epochs, train_acc, 'b--', alpha=0.5, label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r--', alpha=0.5, label='Val Accuracy')
    plt.title('Training & Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fig_02_metrics_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_evaluation_plots(model_path: str, data_path: str, output_dir: str):
    set_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    _, _, test_dataset, _ = create_datasets(data_path, config, test_filepath=data_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
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
    y_bin = np.zeros((y_true.size, n_classes))
    y_bin[np.arange(y_true.size), y_true] = 1
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Test Set Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'fig_03_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_score[:, idx])
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc(fpr, tpr), estimator_name=class_names[idx]).plot(ax=ax, color=color)
    ax.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fig_04_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    auprc_scores = []
    for idx, color in zip(range(n_classes), colors):
        PrecisionRecallDisplay.from_predictions(y_bin[:, idx], y_score[:, idx], name=class_names[idx], color=color, ax=ax)
        auprc_scores.append(average_precision_score(y_bin[:, idx], y_score[:, idx]))
    ax.set_title(f'Precision-Recall Curve (Macro AUPRC: {np.mean(auprc_scores):.4f})')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fig_05_auprc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate main-route training/evaluation plots')
    parser.add_argument('--model', default=os.path.join(MAIN_DIR, 'results_main', 'best_model.pt'))
    parser.add_argument('--data', default=os.path.join(MAIN_DIR, 'Dataset', 'Testing_Dataset', 'Test1.json'))
    parser.add_argument('--training-log', default=None, help='Optional training_log.json or training_log_final.json')
    parser.add_argument('--output', default=os.path.join(MAIN_DIR, 'plots'))
    args = parser.parse_args()
    if args.training_log:
        plot_learning_curves(args.training_log, args.output)
    generate_evaluation_plots(args.model, args.data, args.output)
    print(f'Plots saved to {args.output}')
if __name__ == '__main__':
    main()
