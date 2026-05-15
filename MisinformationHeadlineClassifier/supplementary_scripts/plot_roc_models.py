import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
CLEAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY_DIR = os.path.join(CLEAN_ROOT, 'Binary_model')
BASE_MODEL = 'bert-base-uncased'
MAX_SEQ_LEN = 128
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class BertClassifier(torch.nn.Module):

    def __init__(self, num_classes=2, dropout_rate=0.3, local_files_only=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(BASE_MODEL, local_files_only=local_files_only)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.pooler_output))

class TestDataset(Dataset):

    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), max_length=MAX_SEQ_LEN, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0)}

def get_probabilities(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            probs.extend(torch.softmax(out, dim=-1).cpu().numpy())
    return np.array(probs)

def load_test_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    texts, topics, labels = ([], [], [])
    for row in data:
        if row.get('framework1_feature1') != 1 and row.get('framework1_feature2') != 1:
            continue
        texts.append(row['text'])
        topics.append(row.get('topic', ''))
        labels.append(0 if row.get('framework1_feature1') == 1 else 1)
    return (texts, topics, np.array(labels))

def averaged_fold_probs(weights_dir, test_loader, topics, local_files_only):
    combined, separate = ([], [])
    for fold in range(1, 11):
        model = BertClassifier(local_files_only=local_files_only).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(weights_dir, 'weights_combined', f'fold_{fold:02d}_model.pt'), map_location=DEVICE))
        combined.append(get_probabilities(model, test_loader)[:, 1])
        model_h = BertClassifier(local_files_only=local_files_only).to(DEVICE)
        model_h.load_state_dict(torch.load(os.path.join(weights_dir, 'weights_health', f'fold_{fold:02d}_model.pt'), map_location=DEVICE))
        probs_h = get_probabilities(model_h, test_loader)[:, 1]
        model_t = BertClassifier(local_files_only=local_files_only).to(DEVICE)
        model_t.load_state_dict(torch.load(os.path.join(weights_dir, 'weights_tech', f'fold_{fold:02d}_model.pt'), map_location=DEVICE))
        probs_t = get_probabilities(model_t, test_loader)[:, 1]
        separate.append([probs_h[i] if topics[i] == 'health' else probs_t[i] for i in range(len(topics))])
    return (np.mean(combined, axis=0), np.mean(separate, axis=0))

def main():
    parser = argparse.ArgumentParser(description='Plot ROC curves for binary fold models')
    parser.add_argument('--test-data', default=os.path.join(BINARY_DIR, 'Datasets', 'Test', 'test_dataset_preprocessed.json'))
    parser.add_argument('--weights-dir', default=os.path.join(BINARY_DIR, 'training_results'), help='Directory containing weights_combined/, weights_health/, weights_tech/')
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
    combined_probs, sep_probs = averaged_fold_probs(args.weights_dir, loader, topics, not args.allow_download)
    combined_fpr, combined_tpr, _ = roc_curve(labels, combined_probs)
    sep_fpr, sep_tpr, _ = roc_curve(labels, sep_probs)
    combined_auc = roc_auc_score(labels, combined_probs)
    sep_auc = roc_auc_score(labels, sep_probs)
    plt.figure(figsize=(8, 8))
    plt.plot(combined_fpr, combined_tpr, color='#1976D2', lw=2, label=f'Combined Model (AUC = {combined_auc:.3f})')
    plt.plot(sep_fpr, sep_tpr, color='#F57C00', lw=2, label=f'Separate Models Routed (AUC = {sep_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Baseline (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.output_dir, 'roc_curve_models.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'ROC plot saved to {out}')
if __name__ == '__main__':
    main()
