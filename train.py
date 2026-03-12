import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier


CONFIG = {
    'dataset'      : 'mthvd',
    'w2v_path'     : 'subtrees/mthvd/node_w2v_128',
    'embedding_dim': 128,
    'encode_dim'   : 128,
    'hidden_dim'   : 256,
    'label_size'   : 2,
    'batch_size'   : 32,
    'epochs'       : 100,
    'lr'           : 5e-4,
    'weight_decay' : 1e-5,
    'patience'     : 10,
    'ckpt_dir'     : 'checkpoints',
}


class CodeDataset(Dataset):
    def __init__(self, pkl_path):
        self.data = pd.read_pickle(pkl_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'features': row['features'],
            'target'  : int(row.get('target', 0))
        }


def collate_fn(batch, pad_value=0):
    subtrees_batch = []
    cfg_flat_batch = []
    dfg_batch      = []
    labels         = []

    for item in batch:
        features = item['features']
        subtrees_batch.append(features['subtrees'])

        paths = list(features['cfg_paths'])[:3]
        while len(paths) < 3:
            paths.append([pad_value])

        for p in paths:
            
            p = p if (isinstance(p, list) and len(p) > 0) else [pad_value]
            cfg_flat_batch.append(torch.tensor(p, dtype=torch.long))

        dfg_batch.append(torch.tensor(features['dfg_seqs'], dtype=torch.long))
        labels.append(item['target'])

    cfg_padded       = pad_sequence(cfg_flat_batch, batch_first=True, padding_value=pad_value)
    B                = len(batch)
    max_path_len     = cfg_padded.size(1)
    cfg_batch_tensor = cfg_padded.view(B, 3, max_path_len)

    dfg_batch_tensor = pad_sequence(dfg_batch, batch_first=True, padding_value=pad_value)
    labels_tensor    = torch.tensor(labels, dtype=torch.long)

    return subtrees_batch, cfg_batch_tensor, dfg_batch_tensor, labels_tensor


def compute_metrics(all_preds, all_labels):
    
    n       = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc     = correct / n if n else 0.0

    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))

    if tp + fp + fn == 0:
        print("  ⚠ Warning: no positive samples or predictions, F1 undefined (set to 0).")
        return acc, 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss            = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for subtrees, cfg_paths, dfg_seqs, labels in loader:
            cfg_paths = cfg_paths.to(device)
            dfg_seqs  = dfg_seqs.to(device)
            labels    = labels.to(device)

            logits     = model(subtrees, cfg_paths, dfg_seqs)
            loss       = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss  = total_loss / len(loader) if loader else 0.0
    acc, f1   = compute_metrics(all_preds, all_labels)
    return avg_loss, acc, f1


def train_model(model, train_loader, val_loader, epochs, device, patience, ckpt_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)

    os.makedirs(ckpt_dir, exist_ok=True)
   
    best_ckpt_path = os.path.join(ckpt_dir, f"best_model_{int(time.time())}.pt")

    best_val_f1    = -1.0
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss       = 0.0
        all_preds_train  = []
        all_labels_train = []
       
        correct_train    = 0
        total_train      = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for subtrees, cfg_paths, dfg_seqs, labels in progress_bar:
            cfg_paths = cfg_paths.to(device)
            dfg_seqs  = dfg_seqs.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            logits = model(subtrees, cfg_paths, dfg_seqs)
            loss   = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

            
            all_preds_train.extend(preds.cpu().tolist())
            all_labels_train.extend(labels.cpu().tolist())

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc' : f"{correct_train / total_train:.4f}"
            })

        scheduler.step()

   
        train_acc, train_f1       = compute_metrics(all_preds_train, all_labels_train)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:>3} | "
            f"Train Loss: {total_loss/len(train_loader):.4f}  "
            f"Acc: {train_acc:.4f}  F1: {train_f1:.4f} | "
            f"Val   Loss: {val_loss:.4f}  "
            f"Acc: {val_acc:.4f}  F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ✔ Best model saved → {best_ckpt_path}  (Val F1={val_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print(f"\nBest Val F1: {best_val_f1:.4f}  |  checkpoint: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    
    print("Loading Word2Vec vocab...")
    w2v_model  = Word2Vec.load(CONFIG['w2v_path']).wv
    VOCAB_SIZE = w2v_model.vectors.shape[0] + 1   # +1 为 PAD token 保留
    PAD_VALUE  = VOCAB_SIZE - 1
    print(f"Vocab size: {VOCAB_SIZE}  |  PAD index: {PAD_VALUE}")

    print("Building pretrained embedding matrix...")
    pretrained = np.zeros((VOCAB_SIZE, CONFIG['embedding_dim']), dtype=np.float32)
    for word, idx in w2v_model.key_to_index.items():
        pretrained[idx] = w2v_model[word]


    print("Loading datasets...")
    train_dataset = CodeDataset(f"subtrees/{CONFIG['dataset']}/train_features.pkl")
    val_dataset   = CodeDataset(f"subtrees/{CONFIG['dataset']}/val_features.pkl")
    test_dataset  = CodeDataset(f"subtrees/{CONFIG['dataset']}/test_features.pkl")

    collate = lambda b: collate_fn(b, pad_value=PAD_VALUE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'], shuffle=True,
        drop_last=True,
        collate_fn=collate, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'], shuffle=False,
        drop_last=False,
        collate_fn=collate, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'], shuffle=False,
        drop_last=False,
        collate_fn=collate, num_workers=4, pin_memory=True
    )

    print("Initializing model...")
    model = BatchProgramClassifier(
        embedding_dim    = CONFIG['embedding_dim'],
        hidden_dim       = CONFIG['hidden_dim'],
        vocab_size       = VOCAB_SIZE,
        encode_dim       = CONFIG['encode_dim'],
        label_size       = CONFIG['label_size'],
        batch_size       = CONFIG['batch_size'],
        device           = DEVICE,
        use_gpu          = torch.cuda.is_available(),
        pretrained_weight= pretrained
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    print("Starting training...")
    best_ckpt = train_model(
        model, train_loader, val_loader,
        epochs   = CONFIG['epochs'],
        device   = DEVICE,
        patience = CONFIG['patience'],
        ckpt_dir = CONFIG['ckpt_dir']
    )

    print("\nFinal evaluation on test set...")
    model.load_state_dict(
        torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
    )
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test  Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  F1: {test_f1:.4f}")