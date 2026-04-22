"""
src/augmentation.py — Audio & Spectrogram Augmentation
"""

import numpy as np
import librosa
from collections import Counter


def augment_audio_for_yolo(y, sr):
    """Audio-level augmentation for YOLO. Returns (audio, suffix) pairs."""
    return [
        (y, ""),
        (y + np.random.normal(0, 0.004, len(y)), "_noise"),
        (y * 1.2, "_loud"),
    ]


def oversample_chunks(all_specs, all_labels, h_copies=3, sb_copies=1):
    """Duplicate chunks containing rare classes (h, sb)."""
    augmented_specs = list(all_specs)
    augmented_labels = list(all_labels)

    h_count, sb_count = 0, 0
    for i in range(len(all_specs)):
        labels_in_chunk = set(all_labels[i])

        if 3 in labels_in_chunk:  # harmonic
            for _ in range(h_copies):
                augmented_specs.append(all_specs[i])
                augmented_labels.append(all_labels[i])
            h_count += h_copies

        elif 1 in labels_in_chunk:  # single burst
            for _ in range(sb_copies):
                augmented_specs.append(all_specs[i])
                augmented_labels.append(all_labels[i])
            sb_count += sb_copies

    print(f"  Oversampled: {len(all_specs)} → {len(augmented_specs)} (+{h_count} h, +{sb_count} sb)")
    return augmented_specs, augmented_labels
