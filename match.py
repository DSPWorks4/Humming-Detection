import librosa
import numpy as np
import scipy.ndimage 
import os
import glob

DB_FOLDER = 'database'
SAMPLE_RATE = 22050

def preprocess_chroma(chroma):
    return scipy.ndimage.median_filter(chroma, size=(1, 9))

def get_dtw_distance(hum_chroma, song_chroma):
    # Similar to edit distance problem

    D, wp = librosa.sequence.dtw(hum_chroma, song_chroma, metric='cosine', subseq=True)
    
    # Get distance cost
    best_row, best_col = wp[0] # End point in song
    min_cost = D[best_row, best_col]
    
    path_length = len(wp)
    
    # Stretch Factor
    hum_len = hum_chroma.shape[1]
    song_match_len = abs(wp[0][1] - wp[-1][1]) # Start frame - End frame in song
    
    if hum_len == 0: hum_len = 1
    
    ratio = song_match_len / hum_len
    
    # A perfect match has a ratio near 1.0 (Same speed).
    # If ratio is 0.2 (Matched 10s hum to 2s song part) -> BAD.
    # If ratio is 3.0 (Matched 10s hum to 30s song part) -> BAD.
    
    # Apply Penalty: If ratio is too extreme,to  increase the cost.
    penalty = 1.0
    if ratio < 0.5 or ratio > 2.0:
        penalty = 1.5 
    elif ratio < 0.8 or ratio > 1.2:
        penalty = 1.1 
        
    final_score = (min_cost / path_length) * penalty
    
    return final_score

def match_hum(hum_path):
    print(f"--- Analyze Humming: {hum_path} ---")
    
    try:
        y, sr = librosa.load(hum_path, sr=SAMPLE_RATE, mono=True, duration=30)
        y, _ = librosa.effects.trim(y, top_db=20)
        y = librosa.util.normalize(y)
        
        hum_chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        hum_chroma = preprocess_chroma(hum_chroma)
        
    except Exception as e:
        print(f"Error: {e}")
        return

    npy_files = glob.glob(os.path.join(DB_FOLDER, '*.npy'))
    results = []

    for npy_file in npy_files:
        song_chroma = np.load(npy_file)
        song_name = os.path.splitext(os.path.basename(npy_file))[0]
        
        song_chroma = preprocess_chroma(song_chroma)

        # OPTIMIZATION: Frequency Shift + DTW
        best_dist = float('inf')
        for i in range(12):
            shifted_hum = np.roll(hum_chroma, i, axis=0)
            dist = get_dtw_distance(shifted_hum, song_chroma)
            if dist < best_dist:
                best_dist = dist
        
        results.append((best_dist, song_name))

    # Sort and Decision
    results.sort(key=lambda x: x[0])
    best_score, best_name = results[0]
    second_score, _ = results[1] if len(results) > 1 else (1.0, "None")
    
    print("\n" + "="*30)
    print(f"Top Match: {best_name} (Score: {best_score:.4f})")
    
    if best_score < 0.18:
        print(f"RESULT: ✅ MATCH FOUND: {best_name}")
    else:
        print(f"RESULT: ❌ NO MATCH (Score {best_score:.4f} too high)")

match_hum(r"Tests\MOONDEITY x INTERWORLD  ONE CHANCE  SLOWED  REVERBED.mp3")