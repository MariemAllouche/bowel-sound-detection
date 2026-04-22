# Bowel Sound Detection & Classification — DigeHealth Technical Test

## Goal
Develop a proof-of-concept ML model for identifying bowel sounds in audio data and differentiating between 3 main classes:
- **sb** (Single Burst) — short broadband impulse (<0.1s)
- **mb** (Multiple Burst) — repeated impulses (0.1–1s)
- **h** (Harmonic) — sustained tonal sound (>1s)

The model identifies the **start time**, **end time**, and **type** of each bowel sound.

## Results Summary

| Model | Approach | Event Detection F1 | Classification Acc | mAP@0.5 |
|-------|----------|-------------------|-------------------|---------|
| **YOLOv8s v2** | Spectrogram → Object Detection | **0.767** | **0.925** | **0.711** |
| CRNN v1 | Frame-level SED | 0.717 | 0.911 | — |
| CRNN v2 | + Focal Loss + Oversampling | Training... | Training... | — |
| YOLOv8s v3 | + Multi-scale + Audio Aug | Training... | Training... | — |

## Project Structure

```
bowel_sound_project/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.py                      # Shared paths, constants, hyperparameters
│
├── data/                          # Raw data (not committed to git)
│   ├── 23M74M.wav
│   ├── 23M74M.txt
│   ├── AS_1.wav
│   └── AS_1.txt
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py           # Label parsing, audio loading, feature extraction
│   ├── augmentation.py            # Audio augmentation, chunk oversampling
│   ├── crnn_model.py              # CRNN architecture + Focal Loss + training
│   ├── yolo_utils.py              # Spectrogram→image, bbox conversion, dataset gen
│   ├── evaluation.py              # Event-level IoU matching, metrics
│   └── inference.py               # End-to-end: raw audio → (start, end, class)
│
├── eda/
│   ├── eda.py                     # Exploratory data analysis script
│   └── outputs/                   # Generated charts + features CSV
│       ├── 01_class_distribution.png
│       ├── 02_duration_distribution.png
│       ├── 03_audio_annotations_*.png
│       ├── 05_imbalance_analysis.png
│       └── extracted_features.csv
│
├── models/                        # Saved model weights
│   ├── crnn_v1_model.pth
│   ├── crnn_v2_model.pth
│   ├── yolo_v2_best.pt
│   └── yolo_v3_best.pt
│
├── results/                       # Evaluation outputs per model
│   ├── crnn/
│   │   ├── predictions_23M74M.tsv
│   │   ├── predictions_vs_gt.png
│   │   ├── training_curves.png
│   │   └── results.json
│   ├── yolo/
│   │   ├── predictions_23M74M.tsv
│   │   ├── predictions_vs_gt.png
│   │   └── results.json
│   └── model_comparison.json
│
└── presentation/
    └── results_summary.md         # Key findings for interview
```

## Setup

```bash
pip install -r requirements.txt
```

Place audio files (`.wav`) and label files (`.txt`) in the `data/` folder.

## How to Run

### Step 1: Exploratory Data Analysis
```bash
python eda/eda.py
```

### Step 2: Train Models
```bash
# CRNN (Sound Event Detection) — ~15 min on GPU
python -c "from src.crnn_model import main; main()"

# YOLOv8 (Object Detection on spectrograms) — ~1-2 hours on GPU
python -c "from src.yolo_utils import main; main()"
```

### Step 3: Run Inference on New Audio
```bash
# Using CRNN
python src/inference.py --model crnn --audio data/23M74M.wav

# Using YOLO
python src/inference.py --model yolo --audio data/23M74M.wav

# Compare with ground truth
python src/inference.py --model crnn --audio data/23M74M.wav --labels data/23M74M.txt
```

### Step 4: Evaluate and Compare Models
```bash
python src/evaluation.py --compare
```

## Key Findings

1. **Duration is the most discriminative feature** — sb (<0.1s), mb (0.1–1s), h (>1s) separate almost perfectly by duration alone
2. **YOLOv8 is the best end-to-end solution** — detects and classifies in one shot, achieving 0.767 event F1 and 92.5% classification accuracy
3. **CRNN provides frame-level temporal analysis** — finds 87% of events with 91% classification accuracy, but over-detects (low precision on sb)
4. **Class imbalance is the main challenge** — harmonic events are ~3% of data; oversampling and focal loss help
5. **Multi-scale approach matters** — shorter windows (5s) dramatically improve single burst detection

