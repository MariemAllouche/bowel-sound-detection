"""
src/yolo_utils.py — YOLOv8 Sound Event Detection
==================================================
Convert audio → spectrogram images → YOLO object detection.
Supports multi-scale windows and audio augmentation.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import librosa
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.preprocessing import parse_labels
from src.augmentation import augment_audio_for_yolo
from src.evaluation import evaluate_events

cfg.ensure_dirs()
YOLO_DS_DIR = os.path.join(os.path.dirname(cfg.MODELS_DIR), 'yolo_dataset')


def audio_to_spectrogram_image(y, sr, start, end):
    seg = y[int(start * sr):int(end * sr)]
    S = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128, n_fft=1024, hop_length=256, fmax=4000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB -= S_dB.min()
    if S_dB.max() > 0:
        S_dB = S_dB / S_dB.max() * 255
    return S_dB.astype(np.uint8)


def save_spec_image(arr, path):
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4), dpi=100)
    ax.imshow(arr, aspect='auto', origin='lower', cmap='magma', interpolation='bilinear')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


def temporal_nms(detections, iou_threshold=0.5):
    if not detections:
        return []
    dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep, used = [], set()
    for i, d in enumerate(dets):
        if i in used: continue
        cur = d.copy(); used.add(i)
        for j in range(i + 1, len(dets)):
            if j in used or dets[j]['label'] != cur['label']: continue
            ov_s = max(cur['start'], dets[j]['start'])
            ov_e = min(cur['end'], dets[j]['end'])
            ov = max(0, ov_e - ov_s)
            un = (cur['end'] - cur['start']) + (dets[j]['end'] - dets[j]['start']) - ov
            if un > 0 and ov / un > iou_threshold:
                cur['start'] = min(cur['start'], dets[j]['start'])
                cur['end'] = max(cur['end'], dets[j]['end'])
                cur['confidence'] = max(cur['confidence'], dets[j]['confidence'])
                used.add(j)
        keep.append(cur)
    return sorted(keep, key=lambda x: x['start'])


def generate_dataset(dataset_dir=None):
    """Generate multi-scale YOLO dataset with audio augmentation."""
    if dataset_dir is None:
        dataset_dir = YOLO_DS_DIR

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

    windows = []
    for wav_file, txt_file in cfg.FILES:
        name = Path(wav_file).stem
        y_orig, sr = librosa.load(os.path.join(cfg.DATA_DIR, wav_file), sr=cfg.TARGET_SR)
        labels_df = parse_labels(os.path.join(cfg.DATA_DIR, txt_file))
        print(f"{name}: {len(y_orig)/sr:.1f}s, {len(labels_df)} events")

        for y_aug, suffix in augment_audio_for_yolo(y_orig, sr):
            for win_dur, win_hop in cfg.YOLO_WINDOW_CONFIGS:
                ws, wi = 0.0, 0
                while ws + win_dur <= len(y_aug) / sr:
                    we = ws + win_dur
                    spec = audio_to_spectrogram_image(y_aug, sr, ws, we)
                    evts = labels_df[(labels_df['end'] > ws) & (labels_df['start'] < we)]

                    yolo_lines = []
                    for _, ev in evts.iterrows():
                        cs, ce = max(ev['start'], ws), min(ev['end'], we)
                        if ce <= cs: continue
                        xc = ((cs - ws) + (ce - ws)) / 2 / win_dur
                        bw = (ce - cs) / win_dur
                        if bw < 0.001: continue
                        yolo_lines.append(f"{cfg.CLASS_TO_IDX[ev['label']]} {xc:.6f} 0.5 {bw:.6f} 0.8")

                    windows.append({'name': f"{name}{suffix}_{win_dur:.0f}s_w{wi:04d}",
                                    'spec': spec, 'lines': yolo_lines,
                                    'has_h': any(l.startswith("2 ") for l in yolo_lines)})
                    ws += win_hop; wi += 1

    # Oversample h windows
    h_wins = [w for w in windows if w['has_h']]
    for w in h_wins:
        for c in range(3):
            windows.append({**w, 'name': f"{w['name']}_hc{c}"})

    print(f"Total windows: {len(windows)}")

    # Split & save
    np.random.seed(cfg.RANDOM_STATE)
    idx = np.random.permutation(len(windows))
    nt, nv = int(len(idx) * 0.7), int(len(idx) * 0.15)
    split_map = {}
    for i in idx[:nt]: split_map[i] = 'train'
    for i in idx[nt:nt+nv]: split_map[i] = 'val'
    for i in idx[nt+nv:]: split_map[i] = 'test'

    for i, w in enumerate(windows):
        s = split_map[i]
        save_spec_image(w['spec'], os.path.join(dataset_dir, 'images', s, f"{w['name']}.png"))
        with open(os.path.join(dataset_dir, 'labels', s, f"{w['name']}.txt"), 'w') as f:
            f.write('\n'.join(w['lines']))

    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({'path': dataset_dir, 'train': 'images/train', 'val': 'images/val',
                   'test': 'images/test', 'nc': 3, 'names': cfg.TARGET_CLASSES}, f)

    return yaml_path


def predict_audio_multiscale(model, audio_path, output_dir=None):
    """Multi-scale inference: run YOLO at 3s and 5s, merge with temporal NMS."""
    if output_dir is None:
        output_dir = cfg.RESULTS_DIR

    y, sr = librosa.load(audio_path, sr=cfg.TARGET_SR)
    duration = len(y) / sr
    print(f"Inference on: {Path(audio_path).stem} ({duration:.1f}s)")

    all_dets = []
    temp_img = os.path.join(output_dir, '_temp.png')

    for win_dur, win_hop in cfg.YOLO_WINDOW_CONFIGS:
        ws = 0.0
        while ws + win_dur <= duration:
            spec = audio_to_spectrogram_image(y, sr, ws, ws + win_dur)
            save_spec_image(spec, temp_img)
            results = model.predict(temp_img, imgsz=640, conf=0.15, iou=0.45, device=0, verbose=False)

            for result in results:
                if result.boxes is None: continue
                img_w = result.orig_shape[1]
                for box in result.boxes:
                    x1, _, x2, _ = box.xyxy[0].cpu().numpy()
                    all_dets.append({
                        'start': round(float(ws + (x1 / img_w) * win_dur), 4),
                        'end': round(float(ws + (x2 / img_w) * win_dur), 4),
                        'label': cfg.TARGET_CLASSES[int(box.cls[0])],
                        'confidence': round(float(box.conf[0]), 4),
                    })
            ws += win_hop

    if os.path.exists(temp_img):
        os.remove(temp_img)

    events = temporal_nms(all_dets, iou_threshold=0.4)
    print(f"  Detected: {len(events)} events")
    return events


def main():
    

    print("=" * 60)
    print("YOLOv8 SOUND EVENT DETECTION")
    print("=" * 60)

    yaml_path = generate_dataset()

    model = YOLO('yolov8s.pt')
    results = model.train(
        data=yaml_path, epochs=cfg.YOLO_EPOCHS, imgsz=640, batch=cfg.YOLO_BATCH_SIZE,
        patience=cfg.YOLO_PATIENCE, device=0, project=cfg.RESULTS_DIR,
        name='yolo_train', exist_ok=True,
        degrees=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.2,
        copy_paste=0.1, scale=0.5, translate=0.2, box=10.0, cls=1.0,
        lr0=0.001, lrf=0.01, warmup_epochs=5, verbose=True,
    )

    metrics = model.val(data=yaml_path, split='test', device=0,
                         project=cfg.RESULTS_DIR, name='yolo_eval', exist_ok=True)

    print(f"\nmAP@0.5: {metrics.box.map50:.4f}")

    # Inference
    
    audio_path = os.path.join(cfg.DATA_DIR, "23M74M.wav")
    gt_path = os.path.join(cfg.DATA_DIR, "23M74M.txt")
    events = predict_audio_multiscale(model, audio_path)
    ev_metrics = evaluate_events(events, gt_path)

    # Save
    res = {'mAP50': round(float(metrics.box.map50), 4),
           'event_f1': round(ev_metrics['f1'], 4),
           'cls_acc': round(ev_metrics['cls_acc'], 4)}
    with open(os.path.join(cfg.RESULTS_DIR, 'yolo', 'results.json'), 'w') as f:
        json.dump(res, f, indent=2)

    pd.DataFrame(events).to_csv(
        os.path.join(cfg.RESULTS_DIR, 'yolo', 'predictions_23M74M.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
