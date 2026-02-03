"""
Flask Backend for Query-by-Humming Web Application
With DTW re-ranking for improved accuracy.
"""

import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from services import (
    init_database, process_audio_query,
    get_faiss_index, get_metadata, is_database_loaded
)

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    faiss_index = get_faiss_index()
    return jsonify({
        "status": "healthy",
        "database_loaded": is_database_loaded(),
        "songs_count": faiss_index.ntotal if faiss_index else 0
    })


@app.route('/songs', methods=['GET'])
def get_songs():
    """Return the list of available songs in the database."""
    metadata = get_metadata()
    if metadata is None:
        return jsonify({
            "error": "Database not initialized",
            "songs": []
        }), 503
    
    songs = [
        {
            "id": song["index"],
            "title": song["title"],
            "artist": song["artist"],
            "filename": song["filename"]
        }
        for song in metadata
    ]
    
    return jsonify({
        "songs": songs,
        "total": len(songs)
    })


@app.route('/upload', methods=['POST'])
def upload_audio():
    """
    Accept an audio file and find matching songs.
    
    Returns: JSON with top matches and similarity scores
    """
    if not is_database_loaded():
        return jsonify({
            "error": "Database not initialized. Please run build_database.py first."
        }), 503
    
    # Check if audio file is present
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    _, ext = os.path.splitext(audio_file.filename)
    if not ext:
        ext = '.webm'  # Default for browser recordings
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)
    
    try:
        print(f"Processing uploaded file: {audio_file.filename}")
        k = request.args.get('k', default=5, type=int)
        result = process_audio_query(tmp_path, k=k)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to process audio: {str(e)}"
        }), 500
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/test-random', methods=['GET'])
def test_random():
    """
    Test endpoint that picks a random file from the Tests/ folder.
    """
    import random
    from pathlib import Path
    
    tests_dir = Path("Tests")
    if not tests_dir.exists():
        return jsonify({"error": "Tests directory not found"}), 404
    
    test_files = []
    for ext in ['.wav', '.mp3', '.ogg', '.webm', '.m4a', '.mp4']:
        test_files.extend(tests_dir.glob(f"*{ext}"))
    
    if not test_files:
        return jsonify({"error": "No test files found"}), 404
    
    # Pick random file
    test_file = random.choice(test_files)
    
    try:
        result = process_audio_query(str(test_file), k=5, dtw_threshold=0.25)
        result["test_file"] = test_file.name
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    init_database()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )