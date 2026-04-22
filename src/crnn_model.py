"""
src/crnn_model.py — CRNN Sound Event Detection
================================================
Frame-level detection: predict (silence, sb, mb, h) every 10ms.
Then convert frame predictions → events with start, end, class.

Architecture: CNN (spectral features) → BiGRU (temporal context) → Dense (per-frame class)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from scipy.ndimage import median_filter
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.preprocessing import create_frame_labels, parse_labels
from src.augmentation import oversample_chunks
from src.evaluation import evaluate_events
from src.preprocessing import extract_mel_spectrogram
cfg.ensure_dirs()


# ============================================================
# Dataset
# ============================================================

class SEDDataset(Dataset):
    def __init__(self, specs, labels, augment=False):
        self.specs = [torch.FloatTensor(s).unsqueeze(0) for s in specs]
        self.labels = [torch.LongTensor(l) for l in labels]
        self.augment = augment

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        label = self.labels[idx]

        if self.augment:
            spec = spec.clone()
            f_mask = torch.randint(0, 8, (1,)).item()
            f_start = torch.randint(0, max(1, spec.shape[1] - f_mask), (1,)).item()
            spec[:, f_start:f_start + f_mask, :] = 0

            t_mask = torch.randint(0, 50, (1,)).item()
            t_start = torch.randint(0, max(1, spec.shape[2] - t_mask), (1,)).item()
            spec[:, :, t_start:t_start + t_mask] = 0

        return spec, label


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


# ============================================================
# CRNN Model
# ============================================================

class CRNN(nn.Module):
    """
    CNN extracts spectral features per frame (pools frequency, keeps time).
    BiGRU models temporal context across frames.
    Dense outputs per-frame class logits.
    """
    def __init__(self, n_mels=64, num_classes=4, gru_hidden=256, gru_layers=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(0.2),
        )
        self.rnn = nn.GRU(128 * (n_mels // 8), gru_hidden, gru_layers,
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        b, c, f, t = x.shape
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2).reshape(b, t, -1)
        x, _ = self.rnn(x)
        return self.fc(x)


# ============================================================
# Post-Processing
# ============================================================

def frames_to_events(frame_preds, frame_duration=None, min_duration=0.04):
    if frame_duration is None:
        frame_duration = cfg.FRAME_DURATION
    smoothed = median_filter(frame_preds, size=7)
    events = []
    current_label, start_frame = 0, 0

    for i in range(len(smoothed)):
        if smoothed[i] != current_label:
            if current_label != 0:
                st = start_frame * frame_duration
                et = i * frame_duration
                if et - st >= min_duration:
                    events.append({'start': round(st, 4), 'end': round(et, 4),
                                   'label': cfg.IDX_TO_CLASS_SED[current_label],
                                   'duration': round(et - st, 4)})
            current_label = smoothed[i]
            start_frame = i

    if current_label != 0:
        st = start_frame * frame_duration
        et = len(smoothed) * frame_duration
        if et - st >= min_duration:
            events.append({'start': round(st, 4), 'end': round(et, 4),
                           'label': cfg.IDX_TO_CLASS_SED[current_label],
                           'duration': round(et - st, 4)})
    return events


def merge_close_events(events, max_gap=0.03):
    if not events:
        return events
    merged = [events[0].copy()]
    for ev in events[1:]:
        prev = merged[-1]
        if ev['label'] == prev['label'] and ev['start'] - prev['end'] < max_gap:
            prev['end'] = ev['end']
            prev['duration'] = round(prev['end'] - prev['start'], 4)
        else:
            merged.append(ev.copy())
    return merged


# ============================================================
# Full Audio Inference (Overlapping Chunks)
# ============================================================

def predict_full_audio(audio_path, model, device):
    y, sr = librosa.load(audio_path, sr=cfg.TARGET_SR)
    S_dB = extract_mel_spectrogram(y)
    n_frames = S_dB.shape[1]
    chunk_frames = int(cfg.CRNN_CHUNK_DURATION / cfg.FRAME_DURATION)
    hop_frames = chunk_frames // 2

    pred_sums = np.zeros((n_frames, cfg.NUM_CLASSES_SED), dtype=np.float32)
    pred_counts = np.zeros(n_frames, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_frames - chunk_frames + 1, hop_frames):
            end = start + chunk_frames
            chunk = S_dB[:, start:end]
            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=2).cpu().numpy()[0]
            pred_sums[start:end] += probs
            pred_counts[start:end] += 1

        if pred_counts[-1] == 0:
            rs = max(0, n_frames - chunk_frames)
            chunk = S_dB[:, rs:n_frames]
            if chunk.shape[1] < chunk_frames:
                chunk = np.pad(chunk, ((0, 0), (0, chunk_frames - chunk.shape[1])))
            x = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=2).cpu().numpy()[0]
            al = n_frames - rs
            pred_sums[rs:n_frames] += probs[:al]
            pred_counts[rs:n_frames] += 1

    pred_counts = np.maximum(pred_counts, 1)
    all_preds = (pred_sums / pred_counts[:, None]).argmax(axis=1)

    events = frames_to_events(all_preds)
    events = merge_close_events(events)
    return events, all_preds


# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("CRNN SOUND EVENT DETECTION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    all_specs, all_labels = [], []
    for wav, txt in cfg.FILES:
        specs, labels = create_frame_labels(
            os.path.join(cfg.DATA_DIR, wav),
            os.path.join(cfg.DATA_DIR, txt),
        )
        all_specs.extend(specs)
        all_labels.extend(labels)

    print(f"\nOriginal chunks: {len(all_specs)}")

    # Oversample
    all_specs, all_labels = oversample_chunks(all_specs, all_labels)

    # Split
    np.random.seed(cfg.RANDOM_STATE)
    idx = np.random.permutation(len(all_specs))
    nt = int(len(idx) * 0.7)
    nv = int(len(idx) * 0.15)

    train_ds = SEDDataset([all_specs[i] for i in idx[:nt]],
                           [all_labels[i] for i in idx[:nt]], augment=True)
    val_ds = SEDDataset([all_specs[i] for i in idx[nt:nt+nv]],
                         [all_labels[i] for i in idx[nt:nt+nv]], augment=False)
    test_ds = SEDDataset([all_specs[i] for i in idx[nt+nv:]],
                          [all_labels[i] for i in idx[nt+nv:]], augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.CRNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.CRNN_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=cfg.CRNN_BATCH_SIZE)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    model = CRNN(n_mels=cfg.N_MELS, num_classes=cfg.NUM_CLASSES_SED,
                 gru_hidden=cfg.CRNN_GRU_HIDDEN, gru_layers=cfg.CRNN_GRU_LAYERS).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights
    all_fl = np.concatenate(all_labels)
    cw = 1.0 / (np.bincount(all_fl, minlength=cfg.NUM_CLASSES_SED).astype(float) + 1e-8)
    cw = cw / cw.sum() * cfg.NUM_CLASSES_SED
    cw[1] *= 2.0; cw[3] *= 1.5
    cw = cw / cw.sum() * cfg.NUM_CLASSES_SED

    criterion = FocalLoss(weight=torch.FloatTensor(cw).to(device), gamma=cfg.CRNN_FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.CRNN_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.CRNN_EPOCHS)

    best_f1, patience = 0, 0
    for epoch in range(cfg.CRNN_EPOCHS):
        model.train()
        tl = 0
        for s, l in train_loader:
            s, l = s.to(device), l.to(device)
            optimizer.zero_grad()
            loss = criterion(model(s).reshape(-1, cfg.NUM_CLASSES_SED), l.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        scheduler.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for s, l in val_loader:
                vp.extend(model(s.to(device)).argmax(2).cpu().numpy().flatten())
                vl.extend(l.numpy().flatten())

        vf = f1_score(vl, vp, labels=[1, 2, 3], average='macro')
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:>3d} — Loss: {tl/len(train_loader):.4f} | Val F1: {vf:.4f}")

        if vf > best_f1:
            best_f1 = vf; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.CRNN_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"\nBest Val F1: {best_f1:.4f}")

    # Test
    model.eval()
    tp, tl_arr = [], []
    with torch.no_grad():
        for s, l in test_loader:
            tp.extend(model(s.to(device)).argmax(2).cpu().numpy().flatten())
            tl_arr.extend(l.numpy().flatten())

    print("\nTest Frame-Level Report:")
    print(classification_report(tl_arr, tp, labels=[0,1,2,3],
                                 target_names=['silence','sb','mb','h'], digits=4))
    bowel_f1 = f1_score(tl_arr, tp, labels=[1,2,3], average='macro')
    print(f"Bowel Macro F1: {bowel_f1:.4f}")

    # Save
    model_path = os.path.join(cfg.MODELS_DIR, 'crnn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

    # Event-level on 23M74M
    
    audio_path = os.path.join(cfg.DATA_DIR, "23M74M.wav")
    gt_path = os.path.join(cfg.DATA_DIR, "23M74M.txt")
    events, preds = predict_full_audio(audio_path, model, device)
    print(f"\n23M74M: {len(events)} events detected")
    metrics = evaluate_events(events, gt_path)

    # Save results
    results = {'bowel_f1': round(bowel_f1, 4), 'best_val_f1': round(best_f1, 4),
               'event_f1': round(metrics['f1'], 4), 'cls_acc': round(metrics['cls_acc'], 4)}
    with open(os.path.join(cfg.RESULTS_DIR, 'crnn', 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(events).to_csv(
        os.path.join(cfg.RESULTS_DIR, 'crnn', 'predictions_23M74M.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
