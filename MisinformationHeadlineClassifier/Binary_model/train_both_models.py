import json, math, os, copy, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# using local cache so the same BERT files are used each time
os.environ['HF_HOME'] = os.path.join(_PROJECT_ROOT, '.hf_cache')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
warnings.filterwarnings('ignore', category=FutureWarning)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
DEVICE = get_device()
print(f'Using device: {DEVICE}')
# main settings used for all binary runs
BASE_MODEL = 'bert-base-uncased'
MAX_SEQ_LEN = 128
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 2e-05
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.3
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
PATIENCE = 3
N_FOLDS = 10
RANDOM_ITERS = 100
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Datasets', 'Train', 'training_data_balanced_2.json')
TEXT_COL = 'text'
TOPIC_COL = 'topic'
FEATURE1_COL = 'framework1_feature1'
FEATURE2_COL = 'framework1_feature2'
LABEL2IDX = {'central': 0, 'peripheral': 1}
IDX2LABEL = {0: 'central', 1: 'peripheral'}
CLASS_NAMES = ['central', 'peripheral']
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'training_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

class HeadlineDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = (texts, labels)
        self.tokenizer, self.max_len = (tokenizer, max_len)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # turn one headline into BERT input ids and mask
        enc = self.tokenizer(str(self.texts[idx]), max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

class BertClassifier(nn.Module):

    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(BASE_MODEL)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.pooler_output))

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    criterion, total_loss = (nn.CrossEntropyLoss(), 0.0)
    for batch in loader:
        ids, mask = (batch['input_ids'].to(device), batch['attention_mask'].to(device))
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids, mask), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = ([], [])
    with torch.no_grad():
        for batch in loader:
            ids, mask = (batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds = model(ids, mask).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return (acc, f1, np.array(all_preds), np.array(all_labels))

def train_fold(dataset, train_idx, val_idx, device, fold_num, tag, weights_dir):
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    model = BertClassifier(num_classes=2, dropout_rate=DROPOUT_RATE).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = math.ceil(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    best_val_acc, no_improve, best_state = (0.0, 0, None)
    epoch_log = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        epoch_log.append({'epoch': epoch, 'train_loss': round(train_loss, 6), 'val_acc': round(val_acc, 4), 'val_f1': round(val_f1, 4)})
        print(f'    [{tag}] Fold {fold_num:02d}  Epoch {epoch:02d}/{EPOCHS}  loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}')
        if val_acc > best_val_acc:
            # keep the best fold weights, not just the last epoch
            best_val_acc, no_improve = (val_acc, 0)
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'    [{tag}] Fold {fold_num:02d}  Early stop at epoch {epoch}')
                break
    elapsed = time.time() - t0
    fold_path = os.path.join(weights_dir, f'fold_{fold_num:02d}_model.pt')
    torch.save(best_state, fold_path)
    model.load_state_dict(best_state)
    val_acc, val_f1, val_preds, val_true = evaluate(model, val_loader, device)
    train_acc, train_f1, _, _ = evaluate(model, train_loader, device)
    cm = confusion_matrix(val_true, val_preds, labels=[0, 1])
    fold_record = {'fold': fold_num, 'train_samples': len(train_idx), 'val_samples': len(val_idx), 'best_epoch': epoch_log[np.argmax([e['val_acc'] for e in epoch_log])]['epoch'], 'total_epochs': len(epoch_log), 'elapsed_sec': round(elapsed, 1), 'train_acc': round(train_acc, 4), 'train_f1': round(train_f1, 4), 'val_acc': round(val_acc, 4), 'val_f1': round(val_f1, 4), 'confusion_matrix': cm.tolist(), 'weights_path': fold_path, 'epoch_log': epoch_log}
    print(f'    [{tag}] Fold {fold_num:02d} DONE  acc={val_acc:.4f}  f1={val_f1:.4f}  ({elapsed:.0f}s)  weights → {fold_path}\n')
    return (val_acc, val_f1, val_preds, val_true, best_state, fold_record)

def run_kfold(texts, labels, tokenizer, device, tag='Combined'):
    dataset = HeadlineDataset(texts, labels, tokenizer, MAX_SEQ_LEN)
    labels_arr = np.array(labels)
    tag_slug = tag.lower().replace(' ', '_')
    weights_dir = os.path.join(OUTPUT_DIR, f'weights_{tag_slug}')
    os.makedirs(weights_dir, exist_ok=True)
    # stratified folds keep central/peripheral balance in each split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accs, fold_f1s, all_preds, all_true = ([], [], [], [])
    best_acc, best_state = (0.0, None)
    fold_records = []
    print(f"\n{'=' * 60}")
    print(f'  {tag}  —  {N_FOLDS}-fold cross-validation  ({len(texts)} samples)')
    print(f"{'=' * 60}")
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels_arr), 1):
        acc, f1, preds, true, state, record = train_fold(dataset, train_idx, val_idx, device, fold, tag, weights_dir)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        all_preds.extend(preds)
        all_true.extend(true)
        fold_records.append(record)
        if acc > best_acc:
            best_acc, best_state = (acc, state)
    mean_acc, std_acc = (np.mean(fold_accs), np.std(fold_accs))
    mean_f1, std_f1 = (np.mean(fold_f1s), np.std(fold_f1s))
    print(f'\n  {tag} CV Results:  Acc={mean_acc:.4f}±{std_acc:.4f}  F1={mean_f1:.4f}±{std_f1:.4f}')
    report = classification_report(all_true, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(f'\n  {tag} — Aggregated Classification Report:\n{report}')
    cm = confusion_matrix(all_true, all_preds, labels=[0, 1])
    plot_confusion_matrix(cm, CLASS_NAMES, title=f'{tag} — {N_FOLDS}-Fold CV  (acc={mean_acc:.4f}±{std_acc:.4f})', save_path=os.path.join(OUTPUT_DIR, f'cm_{tag_slug}.png'))
    log = {'model_tag': tag, 'n_folds': N_FOLDS, 'n_samples': len(texts), 'hyperparameters': {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'dropout': DROPOUT_RATE, 'max_grad_norm': MAX_GRAD_NORM, 'warmup_ratio': WARMUP_RATIO, 'patience': PATIENCE, 'max_seq_len': MAX_SEQ_LEN}, 'cv_summary': {'mean_acc': round(mean_acc, 4), 'std_acc': round(std_acc, 4), 'mean_f1': round(mean_f1, 4), 'std_f1': round(std_f1, 4)}, 'aggregated_confusion_matrix': cm.tolist(), 'aggregated_classification_report': report, 'folds': fold_records}
    log_path = os.path.join(OUTPUT_DIR, f'training_log_{tag_slug}.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f'    Training log saved → {log_path}')
    return {'fold_accs': fold_accs, 'fold_f1s': fold_f1s, 'mean_acc': mean_acc, 'std_acc': std_acc, 'mean_f1': mean_f1, 'std_f1': std_f1, 'best_state': best_state, 'weights_dir': weights_dir}

def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'    Saved → {save_path}')

def random_baseline(labels, n_iter=RANDOM_ITERS):
    labels_arr = np.array(labels)
    unique, counts = np.unique(labels_arr, return_counts=True)
    prior = counts / counts.sum()
    rng = np.random.default_rng(seed=42)
    accs, f1s = ([], [])
    # random guesses are matched to the dataset class ratio
    for _ in range(n_iter):
        preds = rng.choice(unique, size=len(labels_arr), p=prior)
        accs.append(accuracy_score(labels_arr, preds))
        f1s.append(f1_score(labels_arr, preds, average='macro', zero_division=0))
    mean_acc, std_acc = (np.mean(accs), np.std(accs))
    mean_f1, std_f1 = (np.mean(f1s), np.std(f1s))
    print(f"\n{'=' * 60}\n  Random Baseline ({n_iter} iterations)\n{'=' * 60}")
    print(f'    Accuracy : {mean_acc:.4f} ± {std_acc:.4f}')
    print(f'    Macro F1 : {mean_f1:.4f} ± {std_f1:.4f}')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(['Random'], [mean_acc], yerr=[std_acc], capsize=8, color='#7986CB', edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Random Baseline — Accuracy')
    axes[0].set_ylim(0, 1)
    axes[1].bar(['Random'], [mean_f1], yerr=[std_f1], capsize=8, color='#EF9A9A', edgecolor='black')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Random Baseline — Macro F1')
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'random_baseline.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f'    Saved → {path}')
    return {'mean_acc': mean_acc, 'std_acc': std_acc, 'mean_f1': mean_f1, 'std_f1': std_f1}

def comparison_plot(combined_res, health_res, tech_res, baseline_res):
    names = ['Random\nBaseline', 'Combined\n(H+T)', 'Health\nOnly', 'Tech\nOnly']
    accs = [baseline_res['mean_acc'], combined_res['mean_acc'], health_res['mean_acc'], tech_res['mean_acc']]
    errs = [baseline_res['std_acc'], combined_res['std_acc'], health_res['std_acc'], tech_res['std_acc']]
    f1s = [baseline_res['mean_f1'], combined_res['mean_f1'], health_res['mean_f1'], tech_res['mean_f1']]
    f1e = [baseline_res['std_f1'], combined_res['std_f1'], health_res['std_f1'], tech_res['std_f1']]
    x, w = (np.arange(len(names)), 0.35)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, accs, w, yerr=errs, label='Accuracy', capsize=6, color='#64B5F6', edgecolor='black')
    ax.bar(x + w / 2, f1s, w, yerr=f1e, label='Macro F1', capsize=6, color='#FFB74D', edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — 10-Fold CV')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f'\n    Saved comparison plot → {path}')

def save_best_model(state, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, 'model.pt'))
    tokenizer.save_pretrained(save_dir)
    print(f'    Overall best model saved → {save_dir}/')
if __name__ == '__main__':
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # keep only the two labels used by the final binary model
    df = df[df.apply(lambda r: r[FEATURE1_COL] == 1 or r[FEATURE2_COL] == 1, axis=1)].reset_index(drop=True)

    def get_label(row):
        return LABEL2IDX['central'] if row[FEATURE1_COL] == 1 else LABEL2IDX['peripheral']
    df['label'] = df.apply(get_label, axis=1)
    all_texts, all_labels = (df[TEXT_COL].tolist(), df['label'].tolist())
    health_df = df[df[TOPIC_COL] == 'health'].reset_index(drop=True)
    tech_df = df[df[TOPIC_COL] == 'technology'].reset_index(drop=True)
    print(f'\nDataset: {len(df)} total  |  Health: {len(health_df)}  |  Tech: {len(tech_df)}')
    print(f'Label distribution: {dict(pd.Series(all_labels).value_counts().sort_index())}')
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    baseline_res = random_baseline(all_labels)
    log_combined = os.path.join(OUTPUT_DIR, 'training_log_combined.json')
    if os.path.exists(log_combined):
        with open(log_combined) as f:
            c = json.load(f)['cv_summary']
        combined_res = {'mean_acc': c['mean_acc'], 'std_acc': c['std_acc'], 'mean_f1': c['mean_f1'], 'std_f1': c['std_f1']}
        print('  Skipping Combined model (already trained). Loaded from log.')
    else:
        # this is the main model: health and tech trained together
        combined_res = run_kfold(all_texts, all_labels, tokenizer, DEVICE, tag='Combined')
        save_best_model(combined_res['best_state'], tokenizer, os.path.join(OUTPUT_DIR, 'two_class_model_combined'))
    log_health = os.path.join(OUTPUT_DIR, 'training_log_health.json')
    if os.path.exists(log_health):
        with open(log_health) as f:
            h = json.load(f)['cv_summary']
        health_res = {'mean_acc': h['mean_acc'], 'std_acc': h['std_acc'], 'mean_f1': h['mean_f1'], 'std_f1': h['std_f1']}
        print('  Skipping Health model (already trained). Loaded from log.')
    else:
        # topic-only models are kept for comparison with the combined model
        health_res = run_kfold(health_df[TEXT_COL].tolist(), health_df['label'].tolist(), tokenizer, DEVICE, tag='Health')
        save_best_model(health_res['best_state'], tokenizer, os.path.join(OUTPUT_DIR, 'two_class_model_health'))
    log_tech = os.path.join(OUTPUT_DIR, 'training_log_tech.json')
    if os.path.exists(log_tech):
        with open(log_tech) as f:
            t = json.load(f)['cv_summary']
        tech_res = {'mean_acc': t['mean_acc'], 'std_acc': t['std_acc'], 'mean_f1': t['mean_f1'], 'std_f1': t['std_f1']}
        print('  Skipping Tech model (already trained). Loaded from log.')
    else:
        tech_res = run_kfold(tech_df[TEXT_COL].tolist(), tech_df['label'].tolist(), tokenizer, DEVICE, tag='Tech')
        save_best_model(tech_res['best_state'], tokenizer, os.path.join(OUTPUT_DIR, 'two_class_model_tech'))
    comparison_plot(combined_res, health_res, tech_res, baseline_res)
    print(f"\n{'=' * 60}\n  FINAL SUMMARY\n{'=' * 60}")
    summary = [('Random Baseline', baseline_res), ('Combined (H+T)', combined_res), ('Health Only', health_res), ('Tech Only', tech_res)]
    print(f"  {'Model':<20s}  {'Accuracy':>16s}  {'Macro F1':>16s}")
    print(f"  {'-' * 20}  {'-' * 16}  {'-' * 16}")
    for name, r in summary:
        print(f"  {name:<20s}  {r['mean_acc']:.4f} ± {r['std_acc']:.4f}  {r['mean_f1']:.4f} ± {r['std_f1']:.4f}")
    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{'Model':<20s}  {'Accuracy':>16s}  {'Macro F1':>16s}\n")
        f.write(f"{'-' * 20}  {'-' * 16}  {'-' * 16}\n")
        for name, r in summary:
            f.write(f"{name:<20s}  {r['mean_acc']:.4f} ± {r['std_acc']:.4f}  {r['mean_f1']:.4f} ± {r['std_f1']:.4f}\n")
    print(f'\n  Summary saved → {summary_path}')
    print('  All outputs in → ' + OUTPUT_DIR)
