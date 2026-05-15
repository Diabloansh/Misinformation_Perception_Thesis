from plot_roc_models import BASE_MODEL, BINARY_DIR, CLEAN_ROOT, TestDataset, averaged_fold_probs, load_test_data
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from transformers import BertTokenizer

def main():
    parser = argparse.ArgumentParser(description='Plot micro-average AUPRC curves for binary fold models')
    parser.add_argument('--test-data', default=os.path.join(BINARY_DIR, 'Datasets', 'Test', 'test_dataset_preprocessed.json'))
    parser.add_argument('--weights-dir', default=os.path.join(BINARY_DIR, 'training_results'))
    parser.add_argument('--output-dir', default=os.path.join(BINARY_DIR, 'test_evaluation_results'))
    parser.add_argument('--hf-home', default=os.path.join(CLEAN_ROOT, '.hf_cache'))
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HOME'] = args.hf_home
    os.environ['TRANSFORMERS_OFFLINE'] = '0' if args.allow_download else '1'
    os.makedirs(args.output_dir, exist_ok=True)
    texts, topics, labels = load_test_data(args.test_data)
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL, local_files_only=not args.allow_download)
    loader = DataLoader(TestDataset(texts, tokenizer), batch_size=16, shuffle=False)
    combined_periph, sep_periph = averaged_fold_probs(args.weights_dir, loader, topics, not args.allow_download)
    combined_probs = np.column_stack([1.0 - combined_periph, combined_periph])
    sep_probs = np.column_stack([1.0 - sep_periph, sep_periph])
    y_true_flat = np.eye(2)[labels].ravel()
    combined_precision, combined_recall, _ = precision_recall_curve(y_true_flat, combined_probs.ravel())
    sep_precision, sep_recall, _ = precision_recall_curve(y_true_flat, sep_probs.ravel())
    combined_ap = average_precision_score(y_true_flat, combined_probs.ravel())
    sep_ap = average_precision_score(y_true_flat, sep_probs.ravel())
    plt.figure(figsize=(9, 7))
    plt.plot(combined_recall, combined_precision, color='#1976D2', lw=2, label=f'Combined Model (AP = {combined_ap:.3f})')
    plt.plot(sep_recall, sep_precision, color='#F57C00', lw=2, label=f'Separate Models Routed (AP = {sep_ap:.3f})')
    plt.axhline(y=0.5, color='gray', linestyle='--', lw=2, label='Random Baseline (AP = 0.500)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Micro-Averaged Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.output_dir, 'auprc_curve_micro.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Micro-AUPRC plot saved to {out}')
if __name__ == '__main__':
    main()
