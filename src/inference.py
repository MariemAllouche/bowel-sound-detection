"""
src/inference.py — End-to-End Inference
========================================
Raw audio → (start_time, end_time, class) predictions.

Usage:
  python src/inference.py --model crnn --audio data/23M74M.wav
  python src/inference.py --model yolo --audio data/23M74M.wav
  python src/inference.py --model crnn --audio data/23M74M.wav --labels data/23M74M.txt
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter
import torch
from ultralytics import YOLO

from src.crnn_model import CRNN, predict_full_audio
from src.evaluation import evaluate_events
from src.yolo_utils import predict_audio_multiscale

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

cfg.ensure_dirs()


def inference_crnn(audio_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(cfg.MODELS_DIR, "crnn_model.pth")

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Train first.")
        return []

    model = CRNN(
        n_mels=cfg.N_MELS,
        num_classes=cfg.NUM_CLASSES_SED,
        gru_hidden=cfg.CRNN_GRU_HIDDEN,
        gru_layers=cfg.CRNN_GRU_LAYERS,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    events, _ = predict_full_audio(audio_path, model, device)
    return events


def inference_yolo(audio_path):

    model_path = os.path.join(cfg.MODELS_DIR, "yolo_best.pt")
    if not os.path.exists(model_path):
        # Try alternative paths
        for alt in ["yolo_v2_best.pt", "yolo_v3_best.pt"]:
            alt_path = os.path.join(cfg.MODELS_DIR, alt)
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            print(f"ERROR: No YOLO model found in {cfg.MODELS_DIR}. Train first.")
            return []

    model = YOLO(model_path)
    events = predict_audio_multiscale(model, audio_path, output_dir=cfg.RESULTS_DIR)
    return events


def main():
    parser = argparse.ArgumentParser(description="Bowel sound inference")
    parser.add_argument("--model", required=True, choices=["crnn", "yolo"])
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--labels", default=None, help="Ground truth for evaluation")
    parser.add_argument("--output", default=None, help="Output TSV path")
    args = parser.parse_args()

    print("=" * 60)
    print(f"INFERENCE — {args.model.upper()}")
    print("=" * 60)

    if args.model == "crnn":
        events = inference_crnn(args.audio)
    else:
        events = inference_yolo(args.audio)

    print(f"\nDetected {len(events)} events:")
    counts = Counter(ev["label"] for ev in events)
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    print(f"\n{'Start':>10s}  {'End':>10s}  {'Class':>6s}")
    print("-" * 30)
    for ev in events[:15]:
        print(f"{ev['start']:10.4f}  {ev['end']:10.4f}  {ev['label']:>6s}")
    if len(events) > 15:
        print(f"... and {len(events) - 15} more")

    # Save
    out_path = args.output or os.path.join(
        cfg.RESULTS_DIR, args.model, f"predictions_{Path(args.audio).stem}.tsv"
    )
    pd.DataFrame(events).to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved: {out_path}")

    # Compare with ground truth
    if args.labels:

        evaluate_events(events, args.labels)


if __name__ == "__main__":
    main()
