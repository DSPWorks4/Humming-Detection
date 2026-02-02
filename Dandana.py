import matplotlib
matplotlib.use('Agg') # Headless mode for Windows

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage # For smoothing
import os
import glob

# --- CONFIGURATION ---
SOURCE_FOLDER = 'songs'
DB_FOLDER = 'database'
SAMPLE_RATE = 22050

def build_database():
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
        audio_files.extend(glob.glob(os.path.join(SOURCE_FOLDER, ext)))

    print(f"Found {len(audio_files)} songs. Rebuilding database with HPSS & Smoothing...")

    for file_path in audio_files:
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"Processing: {base_name}...")

            # 1. Load Audio
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            y, _ = librosa.effects.trim(y, top_db=20)

            # 2. OPTIMIZATION A: Harmonic-Percussive Source Separation (HPSS)
            # We EXTRACT only the Harmonic component (melody/chords).
            # We DISCARD the Percussive component (drums/noise).
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # 3. Extract Chroma from the HARMONIC part only
            chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

            # 4. OPTIMIZATION B: Temporal Smoothing
            # We apply the same filter here that we use in the matcher.
            # This makes the song look "softer," like a human voice.
            chromagram = scipy.ndimage.median_filter(chromagram, size=(1, 9))

            # 5. Save Data
            save_path_npy = os.path.join(DB_FOLDER, f"{base_name}.npy")
            np.save(save_path_npy, chromagram)

            # 6. Save Visualization (Optional, for check)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
            plt.title(f'Harmonic Chromagram: {base_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(DB_FOLDER, f"{base_name}.png"))
            plt.close()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("--- Database Rebuild Complete ---")

if __name__ == "__main__":
    build_database()