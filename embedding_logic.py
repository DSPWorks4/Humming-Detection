"""
Embedding Logic for Query-by-Humming System
Uses basic-pitch to convert audio to MIDI notes, then creates a 128-D pitch histogram.
Uses HPSS for harmonic separation and key normalization.
"""

import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import librosa
import tempfile
import os
import subprocess
import shutil

# Formats that need conversion via ffmpeg
_NEEDS_CONVERSION = {'.webm', '.ogg', '.opus', '.m4a', '.aac', '.wma'}


def convert_to_wav(audio_file: str) -> str:
    """
    Convert audio file to WAV format using ffmpeg.
    
    Args:
        audio_file: Path to the input audio file
        
    Returns:
        Path to the converted WAV file (temp file)
    """
    ext = os.path.splitext(audio_file)[1].lower()
    
    # If already a supported format, return as-is
    if ext in {'.wav', '.mp3', '.flac'}:
        return audio_file
    
    # Create temp WAV file
    tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_wav.close()
    
    try:
        # Use ffmpeg to convert
        result = subprocess.run(
            [
                'ffmpeg', '-y',  # Overwrite output
                '-i', audio_file,
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '22050',  # Sample rate
                '-ac', '1',  # Mono
                tmp_wav.name
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            os.unlink(tmp_wav.name)
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        
        return tmp_wav.name
        
    except FileNotFoundError:
        os.unlink(tmp_wav.name)
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to process webm files.")
    except subprocess.TimeoutExpired:
        os.unlink(tmp_wav.name)
        raise RuntimeError("ffmpeg conversion timed out")


def normalize_pitch_to_median(note_events: list, target_median: int = 60) -> list:
    """
    Normalize note pitches so the median pitch equals the target.
    This provides key invariance.
    
    Args:
        note_events: List of (start_time, end_time, pitch_midi, amplitude, pitch_bend)
        target_median: Target median MIDI pitch (default 60 = middle C)
        
    Returns:
        List of note events with shifted pitches
    """
    if not note_events:
        return note_events
    
    # Extract all pitches weighted by duration
    pitches = []
    for note in note_events:
        start_time, end_time, pitch_midi, amplitude, _ = note
        duration = end_time - start_time
        # Add pitch multiple times based on duration (weighted)
        count = max(1, int(duration * 10))  # 10 samples per second
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
        # Clamp to valid MIDI range
        new_pitch = np.clip(new_pitch, 0, 127)
        normalized_events.append((start_time, end_time, new_pitch, amplitude, pitch_bend))
    
    return normalized_events


def generate_embedding(
    audio_file: str,
    use_hpss: bool = True,
    normalize_key: bool = True,
    target_median_pitch: int = 60,
    max_duration: float = 30.0  # Limit audio length for speed
) -> np.ndarray:
    """
    Generate a 128-dimensional pitch histogram embedding from an audio file.
    
    Args:
        audio_file: Path to the audio file (.wav, .mp3, .webm, etc.)
        use_hpss: Whether to apply HPSS for harmonic separation
        normalize_key: Whether to normalize pitch to a reference (key invariance)
        target_median_pitch: Target median MIDI pitch for normalization (default 60 = C4)
        max_duration: Maximum audio duration in seconds (for speed)
        
    Returns:
        A normalized 128-dimensional numpy array representing pitch occurrences
    """
    converted_file = None
    
    try:
        # Step 0: Convert unsupported formats (webm, etc.) to wav
        ext = os.path.splitext(audio_file)[1].lower()
        if ext in _NEEDS_CONVERSION:
            print(f"  Converting {ext} to wav...")
            converted_file = convert_to_wav(audio_file)
            audio_file = converted_file
        
        # Step 1: Load audio
        y, sr = librosa.load(audio_file, sr=22050, mono=True, duration=max_duration)
        
        # Apply HPSS if requested
        if use_hpss:
            print(f"  Using HPSS for harmonic separation...")
            y_harmonic, _ = librosa.effects.hpss(y)
            y = y_harmonic
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Save to temporary file for basic-pitch
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            import soundfile as sf
            sf.write(tmp_path, y, sr)
        
        try:
            # Run basic-pitch inference
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
        
        # Step 2: Key normalization (pitch shift to reference)
        if normalize_key and note_events:
            note_events = normalize_pitch_to_median(note_events, target_median_pitch)
        
        # Step 3: Create 128-dimensional histogram
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
        # Cleanup converted temp file
        if converted_file and os.path.exists(converted_file):
            os.unlink(converted_file)


def generate_embedding_from_array(
    y: np.ndarray,
    sr: int = 22050,
    use_hpss: bool = True,
    normalize_key: bool = True,
    target_median_pitch: int = 60
) -> np.ndarray:
    """
    Generate embedding from a numpy array of audio samples.
    
    Args:
        y: Audio samples as numpy array
        sr: Sample rate
        use_hpss: Whether to apply HPSS
        normalize_key: Whether to normalize pitch to a reference
        target_median_pitch: Target median MIDI pitch for normalization
        
    Returns:
        A normalized 128-dimensional numpy array
    """
    # Apply HPSS if requested
    if use_hpss:
        y_harmonic, _ = librosa.effects.hpss(y)
        y = y_harmonic
    
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
        import soundfile as sf
        sf.write(tmp_path, y, sr)
    
    try:
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
    
    # Key normalization
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


def extract_chromagram(audio_file: str, use_hpss: bool = True) -> np.ndarray:
    """
    Extract chromagram from an audio file for DTW re-ranking.
    
    Args:
        audio_file: Path to the audio file
        use_hpss: Whether to apply HPSS for harmonic separation
        
    Returns:
        Chromagram numpy array of shape (12, T)
    """
    import scipy.ndimage
    
    converted_file = None
    
    try:
        # Convert unsupported formats (webm, etc.) to wav
        ext = os.path.splitext(audio_file)[1].lower()
        if ext in _NEEDS_CONVERSION:
            converted_file = convert_to_wav(audio_file)
            audio_file = converted_file
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
        
        # Apply HPSS if requested
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
        if converted_file and os.path.exists(converted_file):
            os.unlink(converted_file)


def extract_chromagram_from_array(y: np.ndarray, sr: int = 22050, use_hpss: bool = True) -> np.ndarray:
    """
    Extract chromagram from audio array for DTW re-ranking.
    
    Args:
        y: Audio samples as numpy array
        sr: Sample rate
        use_hpss: Whether to apply HPSS
        
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
    # Test with a sample file
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        embedding = generate_embedding(audio_path)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Non-zero elements: {np.count_nonzero(embedding)}")
        print(f"Top 5 pitches: {np.argsort(embedding)[-5:][::-1]}")
    else:
        print("Usage: python embedding_logic.py <audio_file>")