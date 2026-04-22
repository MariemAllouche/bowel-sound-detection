"""
config.py — Shared configuration for the entire project
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
EDA_OUTPUT_DIR = os.path.join(PROJECT_DIR, "eda", "outputs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

FILES = [("23M74M.wav", "23M74M.txt"), ("AS_1.wav", "AS_1.txt")]

# Audio
TARGET_SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
FMAX = 4000
FRAME_DURATION = HOP_LENGTH / TARGET_SR  # 0.01s

# Labels
LABEL_MAP = {'sb': 'sb', 'b': 'sb', 'sbs': 'sb', 'mb': 'mb', 'h': 'h'}
LABEL_MAP_IDX = {'sb': 1, 'b': 1, 'sbs': 1, 'mb': 2, 'h': 3}
TARGET_CLASSES = ['sb', 'mb', 'h']
CLASS_TO_IDX = {'sb': 0, 'mb': 1, 'h': 2}
IDX_TO_CLASS = {0: 'sb', 1: 'mb', 2: 'h'}
IDX_TO_CLASS_SED = {0: 'silence', 1: 'sb', 2: 'mb', 3: 'h'}
NUM_CLASSES_SED = 4

# CRNN
CRNN_CHUNK_DURATION = 10.0
CRNN_EPOCHS = 80
CRNN_PATIENCE = 15
CRNN_LR = 5e-4
CRNN_BATCH_SIZE = 16
CRNN_GRU_HIDDEN = 256
CRNN_GRU_LAYERS = 3
CRNN_FOCAL_GAMMA = 2.0

# YOLO
YOLO_WINDOW_CONFIGS = [(3.0, 1.5), (5.0, 2.5)]
YOLO_IMG_SIZE = 640
YOLO_EPOCHS = 200
YOLO_BATCH_SIZE = 16
YOLO_PATIENCE = 30

RANDOM_STATE = 42

def ensure_dirs():
    for d in [DATA_DIR, EDA_OUTPUT_DIR, MODELS_DIR,
              os.path.join(RESULTS_DIR, "crnn"),
              os.path.join(RESULTS_DIR, "yolo")]:
        os.makedirs(d, exist_ok=True)
