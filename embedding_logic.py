import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import librosa
import tempfile
import os
import subprocess
import shutil

_DEMUCS_AVAILABLE = None

def is_demucs_available():
    """Check if Demucs is installed and available."""
    global _DEMUCS_AVAILABLE
    if _DEMUCS_AVAILABLE is None:
        try:
            result = subprocess.run(
                ['demucs', '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            _DEMUCS_AVAILABLE = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            _DEMUCS_AVAILABLE = False
    return _DEMUCS_AVAILABLE

def isolate_vocals_demucs(audio_file: str, output_dir: str = None) -> str:
    """
    Use Demucs to isolate vocals from an audio file.
        
    Returns:
        Path to the isolated vocals audio file
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='demucs_')
    
    # Run Demucs with htdemucs_ft model (fine-tuned, better quality)
    try:
        subprocess.run(
            [
                'demucs',
                '--two-stems', 'vocals',  
                '-n', 'htdemucs',        
                '-o', output_dir,
                audio_file
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=600  # 10 minute timeout
        )
    except subprocess.CalledProcessError as e:
        print(f"Demucs error: {e.stderr}")
        raise RuntimeError(f"Demucs failed: {e.stderr}")
    
    # Find the vocals file
    audio_name = os.path.splitext(os.path.basename(audio_file))[0]
    vocals_path = os.path.join(output_dir, 'htdemucs', audio_name, 'vocals.wav')
    
    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Demucs output not found at {vocals_path}")
    
    return vocals_path

def normalize_pitch_to_median(note_events: list, target_median: int = 60) -> list:
    """
    Normalize note pitches so the median pitch equals the target.
    This provides key invariance.
        
    Returns:
        List of note events with shifted pitches
    """
    if not note_events:
        return note_events
    
    pitches = []
    for note in note_events:
        start_time, end_time, pitch_midi, amplitude, _ = note
        duration = end_time - start_time
        # Add pitch multiple times based on duration (weighted)
        count = max(1, int(duration * 40))  # 40 samples per second
        pitches.extend([pitch_midi] * count)
    
    if not pitches:
        return note_events
    
    # Calculate median pitch
    current_median = np.median(pitches)
    shift = target_median - current_median
    
    # Shift all note events
    normalized_events = []
    for note in note_events:
        start_time, end_time, pitch_midi, amplitude, pitch_bend = note
        new_pitch = pitch_midi + shift
        new_pitch = np.clip(new_pitch, 0, 127)
        normalized_events.append((start_time, end_time, new_pitch, amplitude, pitch_bend))
    
    return normalized_events

def generate_embedding(
    audio_file: str,
    use_demucs: bool = True,
    use_hpss: bool = True,
    normalize_key: bool = True,
    target_median_pitch: int = 60,
    max_duration: float = 30.0  # Limit audio length for speed
) -> np.ndarray:
    """
    Generate a 128-dimensional pitch histogram embedding from an audio file.
    
    Returns:
        A normalized 128-dimensional numpy array representing pitch occurrences
    """
    demucs_output_dir = None
    
    try:
        # Vocal isolation
        if use_demucs and is_demucs_available():
            print(f"  Using Demucs for vocal isolation...")
            demucs_output_dir = tempfile.mkdtemp(prefix='demucs_')
            vocals_path = isolate_vocals_demucs(audio_file, demucs_output_dir)
            y, sr = librosa.load(vocals_path, sr=22050, mono=True, duration=max_duration)
        else:
            y, sr = librosa.load(audio_file, sr=22050, mono=True, duration=max_duration)
            
            if use_hpss:
                print(f"  Using HPSS for harmonic separation...")
                y_harmonic, _ = librosa.effects.hpss(y)
                y = y_harmonic
        
        y, _ = librosa.effects.trim(y, top_db=20)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            import soundfile as sf
            sf.write(tmp_path, y, sr)
        
        try:
            # Try Tuning these for a better results
            _, _, note_events = predict(
                tmp_path,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5,
                frame_threshold=0.3,
                minimum_note_length=58,  # ~50ms at 22050Hz
                minimum_frequency=65,    # C2 - typical humming range
                maximum_frequency=2093,  # C7 - upper limit
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        # normalization (pitch shift to reference)
        if normalize_key and note_events:
            note_events = normalize_pitch_to_median(note_events, target_median_pitch)
        
        # Create 128-dimensional histogram
        histogram = np.zeros(128, dtype=np.float32)
        
        for note in note_events:
            start_time, end_time, pitch_midi, amplitude, _ = note
            pitch_idx = int(np.clip(pitch_midi, 0, 127))
            duration = end_time - start_time
            histogram[pitch_idx] += duration * amplitude
        
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
        
        return histogram.astype(np.float32)
        
    finally:
        if demucs_output_dir and os.path.exists(demucs_output_dir):
            shutil.rmtree(demucs_output_dir, ignore_errors=True)

def generate_embedding_from_array(
    y: np.ndarray,
    sr: int = 22050,
    use_hpss: bool = True,
    normalize_key: bool = True,
    target_median_pitch: int = 60
) -> np.ndarray:
    """
    Generate embedding from a numpy array of audio samples.
        
    Returns:
        A normalized 128-dimensional numpy array
    """
    if use_hpss:
        y_harmonic, _ = librosa.effects.hpss(y)
        y = y_harmonic
    
    y, _ = librosa.effects.trim(y, top_db=20)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        import soundfile as sf
        sf.write(tmp_path, y, sr)
    
    try:
        # Try Tuning these for a better results
        _, _, note_events = predict(
            tmp_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,
            minimum_frequency=65,
            maximum_frequency=2093,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # Normalization
    if normalize_key and note_events:
        note_events = normalize_pitch_to_median(note_events, target_median_pitch)
    
    # Create histogram
    histogram = np.zeros(128, dtype=np.float32)
    
    for note in note_events:
        start_time, end_time, pitch_midi, amplitude, _ = note
        pitch_idx = int(np.clip(pitch_midi, 0, 127))
        duration = end_time - start_time
        histogram[pitch_idx] += duration * amplitude
    
    # L2 normalize
    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram = histogram / norm
    
    return histogram.astype(np.float32)

def extract_chromagram(audio_file: str, use_demucs: bool = True, use_hpss: bool = True) -> np.ndarray:
    """
    Extract chromagram from an audio file for DTW re-ranking.
        
    Returns:
        Chromagram numpy array of shape (12, T)
    """
    import scipy.ndimage
    
    demucs_output_dir = None
    
    try:
        # Vocal isolation
        if use_demucs and is_demucs_available():
            demucs_output_dir = tempfile.mkdtemp(prefix='demucs_')
            vocals_path = isolate_vocals_demucs(audio_file, demucs_output_dir)
            y, sr = librosa.load(vocals_path, sr=22050, mono=True)
        else:
            y, sr = librosa.load(audio_file, sr=22050, mono=True)
            if use_hpss:
                y_harmonic, _ = librosa.effects.hpss(y)
                y = y_harmonic
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Extract chromagram
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Temporal smoothing
        chromagram = scipy.ndimage.median_filter(chromagram, size=(1, 9))
        
        return chromagram
        
    finally:
        if demucs_output_dir and os.path.exists(demucs_output_dir):
            shutil.rmtree(demucs_output_dir, ignore_errors=True)

def extract_chromagram_from_array(y: np.ndarray, sr: int = 22050, use_hpss: bool = True) -> np.ndarray:
    """
    Extract chromagram from audio array for DTW re-ranking.

    Returns:
        Chromagram numpy array of shape (12, T)
    """
    import scipy.ndimage
    
    if use_hpss:
        y_harmonic, _ = librosa.effects.hpss(y)
        y = y_harmonic
    
    y, _ = librosa.effects.trim(y, top_db=20)
    
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
    chromagram = scipy.ndimage.median_filter(chromagram, size=(1, 9))
    
    return chromagram


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        embedding = generate_embedding(audio_path)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Non-zero elements: {np.count_nonzero(embedding)}")
        print(f"Top 5 pitches: {np.argsort(embedding)[-5:][::-1]}")
    else:
        print("Usage: python embedding_logic.py <audio_file>")