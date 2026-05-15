import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# test uses the same local BERT cache as training
os.environ['HF_HOME'] = os.path.join(_PROJECT_ROOT, '.hf_cache')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
BASE_MODEL = 'bert-base-uncased'
MAX_SEQ_LEN = 128

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
DEVICE = get_device()

class BertClassifier(torch.nn.Module):

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(BASE_MODEL)
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

def get_predictions(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            out = model(ids, mask)
            preds.extend(out.argmax(dim=-1).cpu().numpy())
    return np.array(preds)
if __name__ == '__main__':
    print(f'Using device: {DEVICE}')
    TEST_DATA = os.path.join(os.path.dirname(__file__), 'Datasets', 'Test', 'test_dataset_preprocessed.json')
    OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_evaluation_results')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TEST_DATA, 'r') as f:
        data = json.load(f)
    texts, topics, labels = ([], [], [])
    for r in data:
        # binary labels: central is 0, everything else here is peripheral
        texts.append(r['text'])
        topics.append(r['topic'])
        labels.append(0 if r['framework1_feature1'] == 1 else 1)
    labels = np.array(labels)
    n_samples = len(labels)
    print(f'\nLoaded Test Set: {n_samples} samples')
    unique, counts = np.unique(labels, return_counts=True)
    prior = counts / counts.sum()
    print(f'Distribution: central={counts[0]} ({prior[0]:.1%}), peripheral={counts[1]} ({prior[1]:.1%})')
    rng = np.random.default_rng(seed=42)
    N_ITER = 100
    # simple baselines to check if BERT is actually doing better
    u_acc, u_f1 = ([], [])
    for _ in range(N_ITER):
        p = rng.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        u_acc.append(accuracy_score(labels, p))
        u_f1.append(f1_score(labels, p, average='macro', zero_division=0))
    s_acc, s_f1 = ([], [])
    for _ in range(N_ITER):
        p = rng.choice(unique, size=n_samples, p=prior)
        s_acc.append(accuracy_score(labels, p))
        s_f1.append(f1_score(labels, p, average='macro', zero_division=0))
    maj_p = np.ones(n_samples, dtype=int)
    maj_acc = accuracy_score(labels, maj_p)
    maj_f1 = f1_score(labels, maj_p, average='macro', zero_division=0)
    results = {'Uniform Random': {'acc': np.mean(u_acc), 'acc_std': np.std(u_acc), 'f1': np.mean(u_f1), 'f1_std': np.std(u_f1)}, 'Stratified Random': {'acc': np.mean(s_acc), 'acc_std': np.std(s_acc), 'f1': np.mean(s_f1), 'f1_std': np.std(s_f1)}, 'Majority (Periph)': {'acc': maj_acc, 'acc_std': 0.0, 'f1': maj_f1, 'f1_std': 0.0}}
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    test_loader = DataLoader(TestDataset(texts, tokenizer), batch_size=16, shuffle=False)
    print('\nEvaluating Combined Model across 10 folds...')
    c_acc, c_f1 = ([], [])
    # average the test score across all saved CV folds
    for fold in range(1, 11):
        model = BertClassifier().to(DEVICE)
        weight_path = os.path.join(os.path.dirname(__file__), 'training_results', 'weights_combined', f'fold_{fold:02d}_model.pt')
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        preds = get_predictions(model, test_loader)
        c_acc.append(accuracy_score(labels, preds))
        c_f1.append(f1_score(labels, preds, average='macro', zero_division=0))
    results['Combined (H+T)'] = {'acc': np.mean(c_acc), 'acc_std': np.std(c_acc), 'f1': np.mean(c_f1), 'f1_std': np.std(c_f1)}
    print('Evaluating Separate Models (Health/Tech) across 10 folds...')
    sep_acc, sep_f1 = ([], [])
    for fold in range(1, 11):
        model_h = BertClassifier().to(DEVICE)
        model_h.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'training_results', 'weights_health', f'fold_{fold:02d}_model.pt'), map_location=DEVICE))
        preds_h = get_predictions(model_h, test_loader)
        model_t = BertClassifier().to(DEVICE)
        model_t.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'training_results', 'weights_tech', f'fold_{fold:02d}_model.pt'), map_location=DEVICE))
        preds_t = get_predictions(model_t, test_loader)
        final_preds = []
        # route by the known topic for this comparison
        for i, topic in enumerate(topics):
            final_preds.append(preds_h[i] if topic == 'health' else preds_t[i])
        sep_acc.append(accuracy_score(labels, final_preds))
        sep_f1.append(f1_score(labels, final_preds, average='macro', zero_division=0))
    results['Separate (H/T routed)'] = {'acc': np.mean(sep_acc), 'acc_std': np.std(sep_acc), 'f1': np.mean(sep_f1), 'f1_std': np.std(sep_f1)}
    print(f"\n{'=' * 60}\n  TEST SET FINAL EVALUATION SUMMARY\n{'=' * 60}")
    print(f"  {'Model Approach':<25s}  {'Accuracy':>16s}  {'Macro F1':>16s}")
    print(f"  {'-' * 25}  {'-' * 16}  {'-' * 16}")
    with open(os.path.join(OUT_DIR, 'test_summary.txt'), 'w') as f:
        f.write(f"{'Model Approach':<25s}  {'Accuracy':>16s}  {'Macro F1':>16s}\n")
        f.write(f"{'-' * 25}  {'-' * 16}  {'-' * 16}\n")
        for name, r in results.items():
            line = f"  {name:<25s}  {r['acc']:.4f} ± {r['acc_std']:.4f}  {r['f1']:.4f} ± {r['f1_std']:.4f}"
            print(line)
            f.write(line + '\n')
    names = list(results.keys())
    accs = [results[n]['acc'] for n in names]
    a_err = [results[n]['acc_std'] for n in names]
    f1s = [results[n]['f1'] for n in names]
    f_err = [results[n]['f1_std'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, accs, w, yerr=a_err, label='Accuracy', capsize=6, color='#64B5F6', edgecolor='black')
    ax.bar(x + w / 2, f1s, w, yerr=f_err, label='Macro F1', capsize=6, color='#FFB74D', edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Test Set Performance: Baselines vs Trained Models')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'test_performance_plot.png'), dpi=150)
    print(f'\nSaved results to {OUT_DIR}/test_summary.txt and test_performance_plot.png')
