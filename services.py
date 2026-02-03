import numpy as np
import librosa
import scipy.ndimage
import os

from embedding_logic import generate_embedding, extract_chromagram_from_array
from build_database import load_database, search, METADATA_FILE, CHROMAGRAM_DIR

# Module-level state
_faiss_index = None
_metadata = None

# Weighted Scoring Configuration
# These weights determine the contribution of each metric to the final score
EMBEDDING_WEIGHT = 0.6  # Higher weight = embeddings more important
DTW_WEIGHT = 0.4        # Higher weight = DTW more important

# Final match threshold (0-1 scale, higher = stricter)
MATCH_THRESHOLD = 0.65  # Tuned to reduce false positives

# Legacy thresholds (kept for backwards compatibility)
SIMILARITY_THRESHOLD = 0.4  
DTW_THRESHOLD = 0.2         

def get_dtw_distance(query_chroma: np.ndarray, song_chroma: np.ndarray) -> float:
    """
    Calculate DTW distance between query chromagram and song chromagram.
    for re-ranking.
        
    Returns:
        DTW distance score (lower is better)
    """

    query_len = query_chroma.shape[1]
    song_len = song_chroma.shape[1]
    
    # Truncate query if it's longer than the song (shouldn't match anyway)
    if query_len > song_len:
        start = (query_len - song_len) // 2
        query_chroma = query_chroma[:, start:start + song_len]
        query_len = song_len
    
    # Ensure we have enough frames for comparison
    if query_len < 10 or song_len < 10:
        return float('inf')
    
    try:
        # subseq=True: finds where query best matches within song
        D, wp = librosa.sequence.dtw(query_chroma, song_chroma, metric='cosine', subseq=True)
        
        best_row, best_col = wp[0]
        min_cost = D[best_row, best_col]
        path_length = len(wp)
        
        song_match_len = abs(wp[0][1] - wp[-1][1]) + 1
        
        if query_len == 0:
            query_len = 1
        
        ratio = song_match_len / query_len
        
        # Apply penalty for extreme differences
        penalty = 1.0
        if ratio < 0.5 or ratio > 2.0:
            penalty = 1.5
        elif ratio < 0.8 or ratio > 1.2:
            penalty = 1.1
        
        final_score = (min_cost / max(path_length, 1)) * penalty
        return final_score
        
    except Exception as e:
        print(f"DTW computation error: {e}")
        return float('inf')

def dtw_rerank(
    query_chroma: np.ndarray,
    candidates: list,
    top_k: int = 5
) -> list:
    """
    Re-rank FAISS candidates using DTW on chromagrams.
    Tests all 12 semitone shifts for key invariance.
        
    Returns:
        Re-ranked list of (metadata, dtw_score, faiss_score) tuples
    """
    results = []
    
    for meta, faiss_score in candidates:
        # Load song chromagram
        chroma_path = meta.get("chromagram_path")
        if not chroma_path or not os.path.exists(chroma_path):
            # Fall back to FAISS score if no chromagram
            results.append((meta, float('inf'), faiss_score))
            continue
        
        try:
            song_chroma = np.load(chroma_path)
            
            # Test all 12 semitone shifts for key invariance
            best_dtw = float('inf')
            for shift in range(12):
                shifted_query = np.roll(query_chroma, shift, axis=0)
                dtw_score = get_dtw_distance(shifted_query, song_chroma)
                best_dtw = min(best_dtw, dtw_score)
            
            results.append((meta, best_dtw, faiss_score))
            
        except Exception as e:
            print(f"DTW error for {meta.get('title', 'unknown')}: {e}")
            results.append((meta, float('inf'), faiss_score))
    
    # Sort by DTW score (lower is better)
    results.sort(key=lambda x: x[1])
    
    return results[:top_k]

def compute_weighted_score(
    embedding_similarity: float,
    dtw_score: float,
    embedding_weight: float = EMBEDDING_WEIGHT,
    dtw_weight: float = DTW_WEIGHT
) -> float:
    """
    Compute weighted combined score from embedding similarity and DTW.
    
    Args:
        embedding_similarity: FAISS similarity score (0-1, higher is better)
        dtw_score: DTW distance (lower is better, typically 0-0.5)
        embedding_weight: Weight for embedding score (default 0.6)
        dtw_weight: Weight for DTW score (default 0.4)
    
    Returns:
        Combined score (0-1, higher is better)
    """
    # Normalize DTW score to 0-1 range (invert so higher is better)
    # Assume DTW scores typically range from 0 to 0.5
    dtw_normalized = max(0.0, min(1.0, 1.0 - (dtw_score / 0.5)))
    
    # Weighted combination
    weighted_score = (
        embedding_weight * embedding_similarity +
        dtw_weight * dtw_normalized
    )
    
    return weighted_score

def rerank_with_weighted_score(
    candidates: list,
    embedding_weight: float = EMBEDDING_WEIGHT,
    dtw_weight: float = DTW_WEIGHT
) -> list:
    """
    Re-rank candidates using weighted score combining embedding + DTW.
    
    Args:
        candidates: List of (metadata, dtw_score, faiss_score) tuples
        
    Returns:
        Re-ranked list sorted by weighted score (highest first)
    """
    scored_candidates = []
    
    for meta, dtw_score, faiss_score in candidates:
        weighted_score = compute_weighted_score(
            faiss_score, dtw_score,
            embedding_weight, dtw_weight
        )
        scored_candidates.append((meta, dtw_score, faiss_score, weighted_score))
    
    # Sort by weighted score (descending - higher is better)
    scored_candidates.sort(key=lambda x: x[3], reverse=True)
    
    return scored_candidates

def init_database():
    """Initialize the database on startup."""
    global _faiss_index, _metadata
    _faiss_index, _metadata = load_database()
    
    if _faiss_index is None:
        print("Warning: Database not found. Run 'python build_database.py' first.")
    else:
        print(f"Database loaded: {_faiss_index.ntotal} songs indexed")


def get_faiss_index():
    """Get the FAISS index."""
    return _faiss_index


def get_metadata():
    """Get the metadata list."""
    return _metadata


def is_database_loaded() -> bool:
    """Check if database is loaded."""
    return _faiss_index is not None

def process_audio_embedding(audio_path: str, max_duration: float = 30.0) -> np.ndarray:
    """
    Generate embedding from audio file.

    Returns:
        128-D embedding vector
    """
    return generate_embedding(
        audio_path,
        use_demucs=False,  # Humming doesn't need Demucs
        use_hpss=True,
        normalize_key=True,
        target_median_pitch=60,
        max_duration=max_duration
    )

def search_by_embedding(query_embedding: np.ndarray, k: int = 15) -> list:
    """
    Search database using FAISS embedding similarity.
        
    Returns:
        List of (metadata, similarity_score) tuples
    """
    return search(query_embedding, k=k)

def filter_candidates(faiss_results: list, threshold: float = SIMILARITY_THRESHOLD) -> list:
    """
    Filter FAISS results by embedding similarity threshold.

    Returns:
        Filtered list of candidates
    """
    return [(meta, score) for meta, score in faiss_results if score > threshold]

def extract_query_chromagram(audio_path: str, max_duration: float = 30.0) -> np.ndarray:
    """
    Extract chromagram from audio file for DTW comparison.
    
    Returns:
        Chromagram array (12, T)
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=max_duration)
    return extract_chromagram_from_array(y, sr, use_hpss=True)

def rerank_with_dtw(query_chroma: np.ndarray, candidates: list, top_k: int = 5) -> list:
    """
    Re-rank candidates using DTW on chromagrams.
        
    Returns:
        Re-ranked list of (metadata, dtw_score, faiss_score) tuples
    """
    return dtw_rerank(query_chroma, candidates, top_k=top_k)

def format_matches(reranked: list, include_filename: bool = True) -> list:
    """
    Format reranked results for JSON response.
        
    Returns:
        List of formatted match dictionaries
    """
    matches = []
    for i, (meta, dtw_score, faiss_score) in enumerate(reranked):
        match = {
            "rank": i + 1,
            "title": meta["title"],
            "artist": meta["artist"],
            "dtw_score": float(round(dtw_score, 4)),
            "similarity": float(round(faiss_score * 100, 2))
        }
        if include_filename:
            match["filename"] = meta["filename"]
        matches.append(match)
    return matches

def process_audio_query(
    audio_path: str,
    k: int = 5,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    dtw_threshold: float = None,  # Optional, uses weighted scoring if None
    match_threshold: float = MATCH_THRESHOLD,
    embedding_weight: float = EMBEDDING_WEIGHT,
    dtw_weight: float = DTW_WEIGHT,
    max_duration: float = 30.0,
    use_weighted_scoring: bool = True
) -> dict:
    """
    Full audio query processing pipeline with weighted scoring.
    
    Args:
        use_weighted_scoring: If True, use weighted combination of scores.
                              If False, use legacy threshold-based approach.
        
    Returns:
        Dictionary with match results
    """
    # STAGE 1: Generate embedding and search
    query_embedding = process_audio_embedding(audio_path, max_duration)
    faiss_results = search_by_embedding(query_embedding, k=min(k * 3, 15))
    
    # Filter by embedding similarity
    filtered_candidates = filter_candidates(faiss_results, similarity_threshold)
    
    if not filtered_candidates:
        return {
            "success": True,
            "found": False,
            "message": f"No match - best embedding similarity {faiss_results[0][1]:.2%} below threshold {similarity_threshold:.0%}",
            "matches": [],
            "best_score": None,
            "threshold": dtw_threshold or match_threshold,
            "similarity_threshold": similarity_threshold
        }
    
    # STAGE 2: Extract chromagram and re-rank with DTW
    query_chroma = extract_query_chromagram(audio_path, max_duration)
    reranked = rerank_with_dtw(query_chroma, filtered_candidates, top_k=k)
    
    # STAGE 3: Apply weighted scoring or threshold-based decision
    if use_weighted_scoring:
        # Use weighted score combining embedding + DTW
        weighted_reranked = rerank_with_weighted_score(
            reranked, embedding_weight, dtw_weight
        )
        
        # Check if best match passes weighted threshold
        best_weighted_score = weighted_reranked[0][3] if weighted_reranked else 0.0
        found_match = bool(best_weighted_score >= match_threshold)
        
        # Format response with weighted scores
        matches = []
        for i, (meta, dtw_score, faiss_score, weighted_score) in enumerate(weighted_reranked):
            matches.append({
                "rank": i + 1,
                "title": meta["title"],
                "artist": meta["artist"],
                "filename": meta["filename"],
                "dtw_score": float(round(dtw_score, 4)),
                "similarity": float(round(faiss_score * 100, 2)),
                "weighted_score": float(round(weighted_score, 4))  # New field
            })
        
        return {
            "success": True,
            "found": found_match,
            "message": "Match found!" if found_match else "No match - weighted score below threshold",
            "matches": matches if found_match else [],
            "best_score": float(round(best_weighted_score, 4)),
            "threshold": match_threshold,
            "similarity_threshold": similarity_threshold,
            "scoring_method": "weighted"
        }
    else:
        # Legacy: Use DTW threshold only
        dtw_threshold = dtw_threshold or DTW_THRESHOLD
        best_dtw_score = reranked[0][1] if reranked else float('inf')
        found_match = bool(best_dtw_score < dtw_threshold)
        
        matches = format_matches(reranked)
        
        return {
            "success": True,
            "found": found_match,
            "message": "Match found!" if found_match else "No match - DTW score exceeds threshold",
            "matches": matches if found_match else [],
            "best_score": float(round(best_dtw_score, 4)),
            "threshold": dtw_threshold,
            "similarity_threshold": similarity_threshold,
            "scoring_method": "threshold"
        }

