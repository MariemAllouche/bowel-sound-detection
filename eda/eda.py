"""
eda/eda.py — Exploratory Data Analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from src.preprocessing import extract_tabular_features

OUT = cfg.EDA_OUTPUT_DIR
os.makedirs(OUT, exist_ok=True)


def parse_all_labels(path):
    """Parse ALL labels including noise/voice for EDA context."""
    events = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    events.append({
                        'start': float(parts[0]), 'end': float(parts[1]),
                        'label': parts[2].strip().lower(),
                    })
                except ValueError:
                    continue
    df = pd.DataFrame(events)
    df['duration'] = df['end'] - df['start']
    return df


def main():
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    all_dfs = []
    for wav_file, txt_file in cfg.FILES:
        name = Path(wav_file).stem
        txt_path = os.path.join(cfg.DATA_DIR, txt_file)
        wav_path = os.path.join(cfg.DATA_DIR, wav_file)

        df = parse_all_labels(txt_path)
        df['source_file'] = name
        all_dfs.append(df)

        print(f"\n{name}: {len(df)} events")
        print(f"  Labels: {sorted(df['label'].unique())}")
        for l, c in df['label'].value_counts().items():
            print(f"    {l:>6s}: {c:>4d} ({c/len(df)*100:.1f}%)")

        if os.path.exists(wav_path):
            info = sf.info(wav_path)
            print(f"  Audio: {info.duration:.1f}s, {info.samplerate}Hz")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Plot 1: Class distribution
    colors_map = {'sb': '#2ecc71', 'b': '#2ecc71', 'mb': '#e67e22', 'h': '#e74c3c',
                  'n': '#95a5a6', 'v': '#3498db', 'sbs': '#27ae60'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (title, df) in enumerate([("Combined", combined)] +
            [(s, combined[combined['source_file'] == s]) for s in combined['source_file'].unique()]):
        if idx >= 3: break
        vc = df['label'].value_counts()
        axes[idx].bar(vc.index, vc.values, color=[colors_map.get(l, '#bdc3c7') for l in vc.index],
                       edgecolor='black', linewidth=0.5)
        axes[idx].set_title(title, fontweight='bold')
        for i, v in enumerate(vc.values):
            axes[idx].text(i, v + 1, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '01_class_distribution.png'), bbox_inches='tight')
    plt.close()

    # Plot 2: Duration distribution
    target = combined[combined['label'].isin(['sb', 'b', 'mb', 'h'])].copy()
    target['label'] = target['label'].replace({'b': 'sb', 'sbs': 'sb'})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cls_colors = {'sb': '#2ecc71', 'mb': '#e67e22', 'h': '#e74c3c'}
    bp_data = [target[target['label'] == l]['duration'].values for l in ['sb', 'mb', 'h']]
    bp = axes[0].boxplot(bp_data, tick_labels=['sb', 'mb', 'h'], patch_artist=True)
    for patch, c in zip(bp['boxes'], cls_colors.values()):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_title('Duration Boxplot', fontweight='bold')
    for l in ['sb', 'mb', 'h']:
        axes[1].hist(target[target['label'] == l]['duration'], bins=20, alpha=0.6,
                      label=l, color=cls_colors[l], edgecolor='black', linewidth=0.3)
    axes[1].set_title('Duration Histogram', fontweight='bold'); axes[1].legend()
    tpc = target.groupby('label')['duration'].sum()
    axes[2].bar(tpc.index, tpc.values, color=[cls_colors[l] for l in tpc.index], edgecolor='black')
    axes[2].set_title('Total Time per Class', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '02_duration_distribution.png'), bbox_inches='tight')
    plt.close()

    # Plot 3: Audio annotations
    for wav_file, txt_file in cfg.FILES:
        name = Path(wav_file).stem
        wav_path = os.path.join(cfg.DATA_DIR, wav_file)
        if not os.path.exists(wav_path): continue
        y, sr = librosa.load(wav_path, sr=22050)
        df = parse_all_labels(os.path.join(cfg.DATA_DIR, txt_file))

        fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
        times = np.arange(len(y)) / sr
        axes[0].plot(times, y, linewidth=0.3, color='#2c3e50', alpha=0.7)
        axes[0].set_title(f'Waveform — {name}', fontweight='bold')
        for _, row in df.iterrows():
            c = colors_map.get(row['label'], '#bdc3c7')
            axes[0].axvspan(row['start'], row['end'], alpha=0.3, color=c)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=4000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axes[1], cmap='magma')
        axes[1].set_title('Mel Spectrogram', fontweight='bold')
        patches = [mpatches.Patch(color=colors_map.get(l, '#bdc3c7'), label=l, alpha=0.6)
                    for l in sorted(df['label'].unique())]
        axes[0].legend(handles=patches, loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f'03_audio_annotations_{name}.png'), bbox_inches='tight')
        plt.close()

    # Plot 5: Imbalance
    counts = target['label'].value_counts()
    mx = counts.max()
    ratios = {l: mx / c for l, c in counts.items()}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ratios.keys(), ratios.values(), color=[cls_colors.get(l, '#bdc3c7') for l in ratios],
            edgecolor='black')
    ax.axhline(1, color='green', ls='--', alpha=0.5)
    ax.set_title('Imbalance Ratio', fontweight='bold')
    for i, (l, v) in enumerate(ratios.items()):
        ax.text(i, v + 0.1, f"{v:.1f}x", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '05_imbalance_analysis.png'), bbox_inches='tight')
    plt.close()

    # Extract features CSV
    print("\nExtracting features...")
    rows = []
    for wav_file, txt_file in cfg.FILES:
        name = Path(wav_file).stem
        wav_path = os.path.join(cfg.DATA_DIR, wav_file)
        if not os.path.exists(wav_path): continue
        y, sr = librosa.load(wav_path, sr=cfg.TARGET_SR)
        labels_df = parse_all_labels(os.path.join(cfg.DATA_DIR, txt_file))
        labels_df = labels_df[labels_df['label'].isin(['sb', 'b', 'mb', 'h', 'sbs'])]
        labels_df['label'] = labels_df['label'].replace({'b': 'sb', 'sbs': 'sb'})

        for _, row in labels_df.iterrows():
            seg = y[int(row['start'] * cfg.TARGET_SR):int(row['end'] * cfg.TARGET_SR)]
            if len(seg) < 160: continue
            feat = extract_tabular_features(seg)
            feat['label'] = row['label']
            feat['source'] = name
            rows.append(feat)

    feat_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, 'extracted_features.csv')
    feat_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({feat_df.shape})")

    print(f"\nAll outputs in: {OUT}")


if __name__ == '__main__':
    main()
