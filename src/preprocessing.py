"""
src/preprocessing.py — Data Loading, Label Parsing, Feature Extraction
"""

import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def parse_labels(path, for_sed=False):
    """Parse Audacity-style label file. Returns list of dicts or DataFrame."""
    events = []
    label_map = cfg.LABEL_MAP_IDX if for_sed else cfg.LABEL_MAP
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    label = parts[2].strip().lower()
                    mapped = label_map.get(label)
                    if mapped:
                        ev = {'start': float(parts[0]), 'end': float(parts[1])}
                        if for_sed:
                            ev['label_idx'] = mapped
                        else:
                            ev['label'] = mapped
                        events.append(ev)
                except ValueError:
                    continue
    return events if for_sed else pd.DataFrame(events)


def load_audio(audio_path):
    """Load and resample audio to TARGET_SR."""
    y, sr = librosa.load(audio_path, sr=cfg.TARGET_SR)
    return y, sr


def extract_mel_spectrogram(audio, normalize=True):
    """Compute mel spectrogram from audio segment."""
    if len(audio) < cfg.N_FFT:
        audio = np.pad(audio, (0, cfg.N_FFT - len(audio)))

    S = librosa.feature.melspectrogram(
        y=audio, sr=cfg.TARGET_SR, n_mels=cfg.N_MELS,
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH, fmax=cfg.FMAX
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    if normalize:
        s_min, s_max = S_dB.min(), S_dB.max()
        if s_max > s_min:
            S_dB = (S_dB - s_min) / (s_max - s_min)
        else:
            S_dB = np.zeros_like(S_dB)

    return S_dB


def extract_tabular_features(audio):
    """Hand-crafted scalar features for EDA and baseline models."""
    sr = cfg.TARGET_SR
    if len(audio) < cfg.N_FFT:
        audio = np.pad(audio, (0, cfg.N_FFT - len(audio)))

    features = {}
    features['duration'] = len(audio) / sr

    rms = librosa.feature.rms(y=audio, frame_length=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    sc = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)[0]
    features['spectral_centroid_mean'] = np.mean(sc)

    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)[0]
    features['spectral_bandwidth_mean'] = np.mean(sb)

    sf = librosa.feature.spectral_flatness(y=audio, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)[0]
    features['spectral_flatness_mean'] = np.mean(sf)

    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)[0]
    features['zcr_mean'] = np.mean(zcr)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH)
    features['chroma_std_mean'] = np.mean(np.std(chroma, axis=1))

    return features


def create_frame_labels(audio_path, label_path, chunk_duration=None):
    """
    Convert audio + annotations → chunks of (spectrogram, frame_labels).
    Each frame gets a label: 0=silence, 1=sb, 2=mb, 3=h.
    """
    if chunk_duration is None:
        chunk_duration = cfg.CRNN_CHUNK_DURATION

    y, sr = load_audio(audio_path)
    events = parse_labels(label_path, for_sed=True)

    S_dB = extract_mel_spectrogram(y, normalize=True)
    n_frames = S_dB.shape[1]

    frame_labels = np.zeros(n_frames, dtype=np.int64)
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=cfg.HOP_LENGTH)

    for ev in events:
        sf = np.searchsorted(frame_times, ev['start'])
        ef = np.searchsorted(frame_times, ev['end'])
        frame_labels[sf:ef] = ev['label_idx']

    chunk_frames = int(chunk_duration / cfg.FRAME_DURATION)
    chunks_spec, chunks_labels = [], []

    for start in range(0, n_frames - chunk_frames, chunk_frames // 2):
        end = start + chunk_frames
        chunks_spec.append(S_dB[:, start:end])
        chunks_labels.append(frame_labels[start:end])

    name = Path(audio_path).stem
    print(f"  {name}: {len(chunks_spec)} chunks, {n_frames} frames, {len(events)} events")
    total = np.concatenate(chunks_labels)
    for idx, name in cfg.IDX_TO_CLASS_SED.items():
        count = np.sum(total == idx)
        print(f"    {name:>8s}: {count:>6d} frames ({count/len(total)*100:.1f}%)")

    return chunks_spec, chunks_labels
