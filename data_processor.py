import os
import librosa
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict

from music.globals import SPECTROGRAM_BIN_DIR, SPECTROGRAM_LABELS_DIR, SOURCE_MUSIC_DIR

# Global constants
STR_TO_DANCE = {
    "cha_cha": ["Cha Cha Cha"],
    "samba": ["Samba"],
    "rumba": ["Rumba"],
    "paso_doble": ["Paso Doble"],
    "jive": ["Jive"],
    "waltz": ["Slow Waltz"],
    "viennese_waltz": ["Viennese Waltz"],
    "tango": ["Tango"],
    "quickstep": ["Quickstep"],
    "foxtrot": ["Slow Foxtrot", "Foxtrot"],
}


# Top-level function for parallel spectrogram generation
def build_spectrogram_window(window_data: Dict, sr: int) -> Dict:
    y = window_data['audio']
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db + 80) / 80

    # Save as .npy file
    np.save(f"{SPECTROGRAM_BIN_DIR}/spectrogram_{window_data['filename']}_window_{window_data['window_index']}.npy", S_db)

    return {
        **window_data,
        'spectrogram_path': f"{SPECTROGRAM_BIN_DIR}/spectrogram_{window_data['filename']}_window_{window_data['window_index']}.npy",
        'shape': S_db.shape,
        'dtype': str(S_db.dtype),
    }


def file_name_to_dance(file_name: str) -> str:
    file_name_lower = file_name.split("-")[-1].lower()
    for dance_key, substrings in STR_TO_DANCE.items():
        for substring in substrings:
            if substring.lower() in file_name_lower:
                return dance_key
    return "unknown"


def process_file(source_dir: str, output_dir: str, filename: str) -> None:
    if not filename.endswith(('.wav', '.mp3')):
        return

    file_path = os.path.join(source_dir, filename)
    try:
        y, sr = librosa.load(file_path, sr=44100)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    dance_label = file_name_to_dance(filename)
    if dance_label == "unknown":
        return

    window_sec = 10
    overlap_sec = 5
    window_samples = window_sec * sr
    step_samples = (window_sec - overlap_sec) * sr

    # Step 1: Split into windows
    windows = []
    for i in range(0, len(y) - window_samples + 1, step_samples):
        window = y[i:i + window_samples]
        if len(window) < window_samples:
            break
        windows.append({
            'window_index': i // step_samples,
            'audio': window,
            'dance_label': dance_label,
            'filename': filename
        })

    # Step 2: Build spectrograms in parallel (within this file)
    with ThreadPoolExecutor() as executor:
        spectrogram_windows = list(
            tqdm(
                executor.map(lambda w: build_spectrogram_window(w, sr), windows),
                total=len(windows),
                desc=f"Building spectrograms for {filename}",
                leave=False
            )
        )

    # Step 3: Save JSON files immediately
    for window_data in spectrogram_windows:
        json_data = {
            'dance': window_data['dance_label'],
            'filename': window_data['filename'],
            'window_index': window_data['window_index'],
            'spectrogram_path': window_data['spectrogram_path'],
            'shape': window_data['shape'],
            'dtype': window_data['dtype'],
        }

        dance = json_data['dance']
        base_name = os.path.splitext(json_data['filename'])[0]
        json_filename = f"{base_name}_window_{json_data['window_index']}.json"
        json_path = os.path.join(output_dir, dance, json_filename)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)


def process_music_sources(source_dir: str, output_dir: str) -> None:
    filenames = [f for f in os.listdir(source_dir) if f.endswith(('.wav', '.mp3'))]

    # Step 0: Process all files in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:
        list(
            tqdm(
                executor.map(lambda f: process_file(source_dir, output_dir, f), filenames),
                total=len(filenames),
                desc="Processing files",
                unit="file"
            )
        )


if __name__ == "__main__":
    process_music_sources(SOURCE_MUSIC_DIR, SPECTROGRAM_LABELS_DIR)
