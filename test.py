import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import Counter

def load_data(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    label_names = data["label_names"].tolist()
    return X, y, label_names

def plot_label_distribution(y):
    counter = Counter(y)
    labels, counts = zip(*sorted(counter.items()))
    plt.figure(figsize=(10, 4))
    plt.bar(labels, counts)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_random_mels(X, y, label_names, n=12):
    indices = random.sample(range(len(X)), n)
    cols = 4
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4 * cols, 3 * rows))

    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[idx], origin='lower', aspect='auto', cmap='magma')
        plt.title(f"{label_names[idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_sample_durations(X, hop_length=512, sr=22050):
    durations = [(x.shape[1] * hop_length) / sr for x in X]

    plt.figure(figsize=(8, 4))
    plt.hist(durations, bins=30, color='teal', edgecolor='black')
    plt.title("Distribution of Sample Durations (s)")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Samples")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"Min duration: {min(durations):.2f}s | Max: {max(durations):.2f}s | Mean: {np.mean(durations):.2f}s")


def main():
    npz_path = "data/cnn_mels_trimmed.npz"
    X, y, label_names = load_data(npz_path)

    print("Loaded dataset:")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Labels shape: {y.shape}")
    print(f"  - Unique labels: {set(label_names)}")

    plot_label_distribution(label_names)
    plot_random_mels(X, y, label_names, n=12)  # Show 12 random mel spectrograms

    plot_sample_durations(X, hop_length=512, sr=22050)


if __name__ == "__main__":
    main()
