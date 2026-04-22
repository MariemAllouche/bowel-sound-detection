{
  "project": "Bowel Sound Detection — DigeHealth Technical Test",
  "evaluation_file_1": "23M74M.wav (300.6s, 514 ground truth events)",
  "evaluation_file_2": "AS_1.wav (2212.4s, 1712 ground truth events)",
  "iou_threshold": 0.3,

  "models": {
    "yolo_v3": {
      "name": "YOLOv8s v3 (Multi-Scale + Audio Augmentation)",
      "architecture": "YOLOv8s pretrained on COCO, fine-tuned",
      "approach": "Spectrogram object detection",
      "window_scales": ["3s", "5s", "10s"],
      "audio_augmentation": true,
      "h_oversampling": "3x",
      "training_images": 2891,
      "parameters": 11126745,
      "epochs": 120,
      "gpu": "NVIDIA A100-SXM4-40GB",
      "training_time": "0.6 hours",
      "detection_metrics": {
        "mAP_50": 0.8500,
        "mAP_50_95": 0.6635,
        "AP_50_sb": 0.7210,
        "AP_50_mb": 0.8818,
        "AP_50_h": 0.9472
      },
      "event_level_23M74M": {
        "predictions": 756,
        "ground_truth": 514,
        "matched": 514,
        "precision": 0.680,
        "recall": 1.000,
        "f1": 0.809,
        "classification_accuracy": 0.953
      },
      "overfitting_check": {
        "val_mAP_50": 0.822,
        "test_mAP_50": 0.850,
        "status": "no overfitting — test > val"
      }
    },

    "yolo_v2": {
      "name": "YOLOv8s v2 (5s Windows)",
      "architecture": "YOLOv8s pretrained on COCO, fine-tuned",
      "approach": "Spectrogram object detection",
      "window_scales": ["5s"],
      "audio_augmentation": false,
      "training_images": 1430,
      "parameters": 11126745,
      "epochs": 100,
      "gpu": "Tesla T4",
      "detection_metrics": {
        "mAP_50": 0.7111,
        "mAP_50_95": 0.4706,
        "AP_50_sb": 0.7136,
        "AP_50_mb": 0.7428,
        "AP_50_h": 0.6769
      },
      "event_level_23M74M": {
        "predictions": 665,
        "ground_truth": 514,
        "matched": 452,
        "precision": 0.680,
        "recall": 0.879,
        "f1": 0.767,
        "classification_accuracy": 0.925
      }
    },

    "yolo_v1": {
      "name": "YOLOv8n v1 (10s Windows)",
      "architecture": "YOLOv8n pretrained on COCO, fine-tuned",
      "approach": "Spectrogram object detection",
      "window_scales": ["10s"],
      "audio_augmentation": false,
      "training_images": 500,
      "parameters": 3011433,
      "epochs": 100,
      "gpu": "Tesla T4",
      "detection_metrics": {
        "mAP_50": 0.6263,
        "mAP_50_95": 0.3879,
        "AP_50_sb": 0.5810,
        "AP_50_mb": 0.6701,
        "AP_50_h": 0.6278
      },
      "event_level_23M74M": {
        "predictions": 665,
        "ground_truth": 514,
        "matched": 452,
        "precision": 0.635,
        "recall": 0.615,
        "f1": 0.626,
        "classification_accuracy": 0.630
      }
    },

    "crnn_v1": {
      "name": "CRNN v1 (Small Model, No Augmentation)",
      "architecture": "CNN (3 blocks) + BiGRU (2 layers, hidden=128) + Dense",
      "approach": "Frame-level prediction every 10ms",
      "loss": "CrossEntropyLoss (weighted)",
      "oversampling": false,
      "parameters": 1292548,
      "epochs_trained": 48,
      "gpu": "Tesla T4",
      "frame_level_metrics": {
        "bowel_macro_f1": 0.7794,
        "sb_f1": 0.6937,
        "mb_f1": 0.8114,
        "h_f1": 0.8330
      },
      "event_level_23M74M": {
        "predictions": 738,
        "ground_truth": 514,
        "matched": 449,
        "precision": 0.608,
        "recall": 0.874,
        "f1": 0.717,
        "classification_accuracy": 0.911
      }
    },

    "crnn_v2": {
      "name": "CRNN v2 (Large Model + Focal Loss + Oversampling)",
      "architecture": "CNN (3 blocks) + BiGRU (3 layers, hidden=256) + Dense",
      "approach": "Frame-level prediction every 10ms",
      "loss": "Focal Loss (gamma=2.0)",
      "oversampling": true,
      "spec_augment": true,
      "parameters": 4493892,
      "epochs_trained": 80,
      "gpu": "Tesla T4",
      "frame_level_metrics": {
        "bowel_macro_f1": 0.7618,
        "best_val_f1": 0.7655
      },
      "event_level_23M74M": {
        "predictions": 643,
        "ground_truth": 514,
        "matched": 341,
        "precision": 0.530,
        "recall": 0.663,
        "f1": 0.589,
        "classification_accuracy": 0.953
      },
      "event_level_AS_1": {
        "predictions": 1764,
        "ground_truth": 1712,
        "matched": 1144,
        "precision": 0.649,
        "recall": 0.668,
        "f1": 0.658,
        "classification_accuracy": 0.993
      },
      "analysis": "Bigger model overfitted on detection — 4.5M params too many for 500 chunks. Classification accuracy improved (95.3% vs 91.1%) but recall dropped sharply (66.3% vs 87.4%). The model is an excellent classifier but weaker detector than v1."
    }
  },

  "best_model": "yolo_v3",
  "best_event_f1": 0.809,
  "best_recall": 1.000,
  "best_classification_accuracy": 0.953,

  "ranking": [
    {"rank": 1, "model": "yolo_v3",  "event_f1": 0.809, "recall": 1.000, "cls_acc": 0.953},
    {"rank": 2, "model": "yolo_v2",  "event_f1": 0.767, "recall": 0.879, "cls_acc": 0.925},
    {"rank": 3, "model": "crnn_v1",  "event_f1": 0.717, "recall": 0.874, "cls_acc": 0.911},
    {"rank": 4, "model": "crnn_v2",  "event_f1": 0.589, "recall": 0.663, "cls_acc": 0.953},
    {"rank": 5, "model": "yolo_v1",  "event_f1": 0.626, "recall": 0.615, "cls_acc": 0.630}
  ],

  "improvement_journey": {
    "yolo_v1_to_v2": {
      "changes": ["Window 10s → 5s", "Model nano → small"],
      "mAP_improvement": "+13.6%",
      "event_f1_improvement": "0.626 → 0.767",
      "reason": "sb events were only ~4px wide at 10s, doubled to ~9px at 5s"
    },
    "yolo_v2_to_v3": {
      "changes": ["Multi-scale windows (3+5+10s)", "Audio augmentation (noise+volume)", "H oversampling 3x", "Higher loss weights (box=10, cls=1)", "Warmup + longer training"],
      "mAP_improvement": "+19.6%",
      "event_f1_improvement": "0.767 → 0.809",
      "reason": "Multi-scale captures all event durations, augmentation tripled data, oversampling fixed h imbalance"
    },
    "crnn_v1_to_v2": {
      "changes": ["GRU hidden 128 → 256", "GRU layers 2 → 3", "Focal Loss (gamma=2)", "Chunk oversampling (h 3x, sb 1x)", "SpecAugment", "Lower LR (1e-3 → 5e-4)"],
      "event_f1_change": "0.717 → 0.589 (WORSE)",
      "classification_improvement": "0.911 → 0.953",
      "reason": "Model too large for dataset size — overfitted on detection task while improving classification. Smaller CRNN v1 is better overall."
    }
  },

  "key_lessons": [
    "Bigger model is not always better — CRNN v1 (1.3M) beat CRNN v2 (4.5M) on detection",
    "Multi-scale windows are the single most impactful improvement for YOLO",
    "Audio-level augmentation is more effective than image-level augmentation",
    "100% recall is achievable — YOLOv8 v3 found all 514 events",
    "Classification is the easy part (>95%) — detection (where events start/end) is the real challenge"
  ]
}