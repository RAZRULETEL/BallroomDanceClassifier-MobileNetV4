import os
import librosa
import numpy as np
import torch
from mobilenetv4 import mobilenetv4_conv_small_grayscale
from globals import DANCES_LABELS, MODELS_DIR
import matplotlib.pyplot as plt
import argparse


# --- Load Model ---
def load_model(model_path=f"{MODELS_DIR}/mobilenetv4_dance_classifier.pth", num_classes=10):
    """Load trained model from disk"""
    model = mobilenetv4_conv_small_grayscale(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# --- Process Song ---
def process_song(song_path, model, dance_to_idx, window_sec=10, overlap_sec=5, sr=44100):
    """Split song into windows, generate spectrograms, and evaluate with model"""
    # Load audio
    y, _ = librosa.load(song_path, sr=sr)

    # Split into windows
    window_samples = window_sec * sr
    step_samples = (window_sec - overlap_sec) * sr

    windows = []
    for i in range(0, len(y) - window_samples + 1, step_samples):
        window = y[i:i + window_samples]
        if len(window) < window_samples:
            break
        windows.append((i, window))

    print(f"Split to {len(windows)} windows")

    # Initialize results storage
    window_results = []
    cumulative_probs = [np.zeros(len(dance_to_idx))]

    # Evaluate each window
    for idx, (start_sample, audio_window) in enumerate(windows):
        # Generate spectrogram
        S = librosa.feature.melspectrogram(
            y=audio_window, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db + 80) / 80  # Normalize

        # Convert to tensor
        spectrogram_tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0).float()  # [1, 1, 128, T]

        # Model prediction
        with torch.no_grad():
            output = model(spectrogram_tensor)
            probs = torch.softmax(output, dim=1).squeeze().numpy()

        # Store results
        window_results.append({
            'window_index': idx,
            'start_time': start_sample / sr,
            'probs': probs.copy()
        })

        # Update cumulative average
        cumulative_probs.append((cumulative_probs[idx] * idx + probs) / (idx + 1))

        # Print per-window result
        print(
            f"\nWindow {idx + 1}/{len(windows)} ({start_sample / sr:.2f}s - {(start_sample + window_samples) / sr:.2f}s):")
        for dance_name, prob in zip(dance_to_idx.keys(), probs):
            print(f"  {dance_name}: {prob:.2%}")

    return window_results, cumulative_probs


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Predict dance type from a song.")
    parser.add_argument("--model", type=str, default="mobilenetv4_dance_classifier_noised_v2.pth",
                        help="Model filename (default: mobilenetv4_dance_classifier_noised_v2.pth)")
    parser.add_argument("--song", type=str, default=None,
                        help="Path to the song")
    args = parser.parse_args()

    # Validate model and song paths
    model_path = os.path.join(MODELS_DIR, args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(args.song):
        raise FileNotFoundError(f"Song file not found: {args.song}")

    # Create dance-to-index mapping
    dance_to_idx = {dance: idx for idx, dance in enumerate(DANCES_LABELS)}

    # Load model
    model = load_model(model_path, num_classes=len(dance_to_idx))

    # Process song
    window_results, cumulative_probs = process_song(args.song, model, dance_to_idx)

    # Print final cumulative results
    print("\nCumulative Average Across All Windows:")
    for dance_name, prob in zip(dance_to_idx.keys(), cumulative_probs[-1]):
        print(f"  {dance_name}: {prob:.2%}")

    # 1. Per-window probabilities
    plt.figure(figsize=(12, 6))
    for i, dance_name in enumerate(dance_to_idx.keys()):
        probs = [res['probs'][i] for res in window_results]
        plt.plot(probs, label=dance_name)
    plt.title("Dance Probability per Window")
    plt.xlabel("Window Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()

    # 2. Cumulative average
    plt.figure(figsize=(12, 6))
    for i, dance_name in enumerate(dance_to_idx.keys()):
        probs = [res[i] for res in cumulative_probs]
        plt.plot(probs, label=dance_name, linestyle='--')
    plt.title("Cumulative Average Probability")
    plt.xlabel("Window Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()

    plt.show()
