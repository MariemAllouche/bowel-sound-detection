"""
src/evaluation.py — Event-Level Evaluation
============================================
Match predictions to ground truth using temporal IoU.
Compute detection precision/recall/F1 and classification accuracy.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.preprocessing import parse_labels


def evaluate_events(predictions, ground_truth_path, iou_threshold=0.3):
    """
    Event-level evaluation using temporal IoU matching.

    For each prediction, find the best-matching ground truth event.
    A match requires IoU ≥ threshold.

    Returns dict with precision, recall, f1, cls_acc.
    """
    gt_events = parse_labels(ground_truth_path, for_sed=True)

    matched_gt = set()
    tp_detect = 0
    tp_class = 0

    for pred in predictions:
        best_iou, best_j = 0, -1
        for j, gt in enumerate(gt_events):
            if j in matched_gt:
                continue
            ov_s = max(pred['start'], gt['start'])
            ov_e = min(pred['end'], gt['end'])
            overlap = max(0, ov_e - ov_s)
            union = (pred['end'] - pred['start']) + (gt['end'] - gt['start']) - overlap
            iou = overlap / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            tp_detect += 1
            gt_label = cfg.IDX_TO_CLASS_SED[gt_events[best_j]['label_idx']]
            if pred['label'] == gt_label:
                tp_class += 1

    n_pred = len(predictions)
    n_gt = len(gt_events)
    precision = tp_detect / n_pred if n_pred > 0 else 0
    recall = tp_detect / n_gt if n_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    cls_acc = tp_class / tp_detect if tp_detect > 0 else 0

    print(f"\n  Event-Level Evaluation (IoU ≥ {iou_threshold}):")
    print(f"    Predictions: {n_pred} | Ground Truth: {n_gt} | Matched: {tp_detect}")
    print(f"    Detection  — P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
    print(f"    Classification accuracy (of matched): {cls_acc:.3f}")

    return {'precision': precision, 'recall': recall, 'f1': f1, 'cls_acc': cls_acc}


def compare_models():
    """Load results from both models and print comparison."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for model_name in ['crnn', 'yolo']:
        res_path = os.path.join(cfg.RESULTS_DIR, model_name, 'results.json')
        if os.path.exists(res_path):
            with open(res_path) as f:
                res = json.load(f)
            print(f"\n  {model_name.upper()}:")
            for k, v in res.items():
                print(f"    {k}: {v}")
        else:
            print(f"\n  {model_name.upper()}: no results found")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()

    if args.compare:
        compare_models()
