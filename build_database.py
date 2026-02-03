"""
Database Builder for Query-by-Humming System
Processes songs from the songs/ folder and builds a FAISS index.
Also stores chromagrams for DTW re-ranking.
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from embedding_logic import generate_embedding, extract_chromagram

# Configuration
SONGS_DIR = "songs"
DATABASE_DIR = "embeddings_db"
INDEX_FILE = os.path.join(DATABASE_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(DATABASE_DIR, "metadata.json")
CHROMAGRAM_DIR = os.path.join(DATABASE_DIR, "chromagrams")
SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.webm'}


def get_song_files(songs_dir: str) -> list:
    """Get all supported audio files from the songs directory."""
    songs_path = Path(songs_dir)
    if not songs_path.exists():
        print(f"Warning: Songs directory '{songs_dir}' does not exist.")
        return []
    
    files = []
    for ext in SUPPORTED_FORMATS:
        files.extend(songs_path.glob(f"*{ext}"))
    
    # Remove duplicates (Windows is case-insensitive)
    seen = set()
    unique_files = []
    for f in files:
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            unique_files.append(f)
    
    return sorted(unique_files)


def extract_song_info(filepath: Path) -> dict:
    """Extract song metadata from filename."""
    # Try to parse "Artist - Title" format
    name = filepath.stem
    if " - " in name:
        parts = name.split(" - ", 1)
        return {
            "artist": parts[0].strip(),
            "title": parts[1].strip(),
            "filename": filepath.name,
            "path": str(filepath)
        }
    else:
        return {
            "artist": "Unknown",
            "title": name,
            "filename": filepath.name,
            "path": str(filepath)
        }


def build_database(songs_dir: str = SONGS_DIR, force_rebuild: bool = False):
    """
    Build the FAISS index from songs in the specified directory.
    Also extracts and stores chromagrams for DTW re-ranking.
    Uses HPSS for harmonic/percussive separation (no Demucs).
    
    Args:
        songs_dir: Directory containing song files
        force_rebuild: If True, rebuild even if database exists
    """
    # Create database directories
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(CHROMAGRAM_DIR, exist_ok=True)
    
    # Check if database already exists
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE) and not force_rebuild:
        print(f"Database already exists at {DATABASE_DIR}")
        print("Use --force to rebuild.")
        return
    
    # Get song files
    song_files = get_song_files(songs_dir)
    
    if not song_files:
        print(f"No audio files found in '{songs_dir}'")
        print(f"Supported formats: {SUPPORTED_FORMATS}")
        return
    
    print(f"Found {len(song_files)} songs to process")
    print("Using HPSS for harmonic separation")
    
    # Process each song
    embeddings = []
    metadata = []
    
    for i, filepath in enumerate(song_files):
        print(f"[{i+1}/{len(song_files)}] Processing: {filepath.name}")
        
        try:
            # Generate embedding with HPSS and key normalization
            embedding = generate_embedding(
                str(filepath),
                use_hpss=True,
                normalize_key=True,
                target_median_pitch=60
            )
            embeddings.append(embedding)
            
            # Extract and save chromagram for DTW re-ranking
            try:
                chromagram = extract_chromagram(
                    str(filepath),
                    use_hpss=True
                )
                chroma_path = os.path.join(CHROMAGRAM_DIR, f"{filepath.stem}.npy")
                np.save(chroma_path, chromagram)
                print(f"  ✓ Saved chromagram: {chromagram.shape}")
            except Exception as chroma_err:
                print(f"  ⚠ Warning: Could not extract chromagram: {chroma_err}")
            
            # Extract metadata
            info = extract_song_info(filepath)
            info["index"] = len(metadata)
            info["chromagram_path"] = os.path.join(CHROMAGRAM_DIR, f"{filepath.stem}.npy")
            metadata.append(info)
            
            print(f"  ✓ Generated embedding (non-zero: {np.count_nonzero(embedding)})")
            
        except Exception as e:
            print(f"  ✗ Error processing {filepath.name}: {e}")
            continue
    
    if not embeddings:
        print("No embeddings generated. Please check your audio files.")
        return
    
    # Stack embeddings into matrix
    embeddings_matrix = np.vstack(embeddings).astype('float32')
    print(f"\nEmbeddings matrix shape: {embeddings_matrix.shape}")
    
    # Build FAISS index (Inner Product for cosine similarity with normalized vectors)
    dimension = 128
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    
    # Save index
    faiss.write_index(index, INDEX_FILE)
    print(f"Index saved to: {INDEX_FILE}")
    
    # Save metadata
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {METADATA_FILE}")
    
    print("\n✓ Database build complete!")
    print(f"  Total songs indexed: {len(metadata)}")


def load_database():
    """
    Load the FAISS index and metadata.
    
    Returns:
        tuple: (faiss_index, metadata_list) or (None, None) if not found
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, None
    
    index = faiss.read_index(INDEX_FILE)
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return index, metadata


def search(query_embedding: np.ndarray, k: int = 5):
    """
    Search for similar songs given a query embedding.
    
    Args:
        query_embedding: 128-dimensional normalized embedding
        k: Number of results to return
        
    Returns:
        List of (metadata, similarity_score) tuples
    """
    index, metadata = load_database()
    
    if index is None:
        raise RuntimeError("Database not found. Run build_database() first.")
    
    # Ensure proper shape
    query = query_embedding.reshape(1, -1).astype('float32')
    
    # Search
    similarities, indices = index.search(query, k)
    
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx >= 0 and idx < len(metadata):
            results.append((metadata[idx], float(sim)))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build song database for QbH system")
    parser.add_argument("--songs-dir", default=SONGS_DIR, help="Directory containing songs")
    parser.add_argument("--force", action="store_true", help="Force rebuild of database")
    
    args = parser.parse_args()
    
    build_database(args.songs_dir, force_rebuild=args.force)