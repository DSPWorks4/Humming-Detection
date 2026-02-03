import React, { useState, useEffect, useRef } from 'react';
import Recorder from './Recorder';
import './App.css';

const API_URL = 'http://localhost:5000';

function App() {
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [songs, setSongs] = useState([]);
    const [testFile, setTestFile] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const fileInputRef = useRef(null);

    // Fetch available songs on mount
    useEffect(() => {
        fetch(`${API_URL}/songs`)
            .then(res => res.json())
            .then(data => setSongs(data.songs || []))
            .catch(err => console.error('Failed to fetch songs:', err));
    }, []);

    const handleRecordingComplete = async (audioBlob) => {
        setIsLoading(true);
        setError(null);
        setResults([]);
        setTestFile(null);

        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        try {
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                if (data.found) {
                    setResults(data.matches);
                } else {
                    setError('No match has been found');
                    setResults([]);
                }
            } else {
                setError(data.error || 'Unknown error occurred');
            }
        } catch (err) {
            setError(`Failed to connect to server: ${err.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleRandomTest = async () => {
        setIsLoading(true);
        setError(null);
        setResults([]);

        try {
            const response = await fetch(`${API_URL}/test-random`);
            const data = await response.json();

            if (data.success) {
                setTestFile(data.test_file);
                if (data.found) {
                    setResults(data.matches);
                    console.log('Random test file matched:', data.test_file);
                } else {
                    console.log('Random test - No match for:', data.test_file);
                    console.log('Best score:', data.best_score, 'Threshold:', data.threshold);
                    setError('No match has been found');
                    setResults([]);
                }
            } else {
                setError(data.error || 'Unknown error occurred');
            }
        } catch (err) {
            setError(`Failed to connect to server: ${err.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
    };

    const handleFileUpload = async () => {
        if (!selectedFile) {
            setError('Please select a file first');
            return;
        }

        setIsLoading(true);
        setError(null);
        setResults([]);
        setTestFile(null);

        const formData = new FormData();
        formData.append('audio', selectedFile);

        try {
            const response = await fetch(`${API_URL}/upload`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                if (data.found) {
                    setResults(data.matches);
                } else {
                    setError('No match has been found');
                    setResults([]);
                }
            } else {
                setError(data.error || 'Unknown error occurred');
            }
        } catch (err) {
            setError(`Failed to connect to server: ${err.message}`);
        } finally {
            setIsLoading(false);
            setSelectedFile(null);
            // Reset file input
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>🎵 Hum-to-Search</h1>
                <p>Hum a tune and find the song!</p>
            </header>

            <main>
                <section className="recorder-section">
                    <Recorder onRecordingComplete={handleRecordingComplete} />

                    <div className="divider">
                        <span>OR</span>
                    </div>

                    <div className="upload-section">
                        <input
                            type="file"
                            accept="audio/*,.mp3,.wav,.webm,.ogg,.m4a"
                            onChange={handleFileSelect}
                            className="file-input"
                            id="file-upload"
                            ref={fileInputRef}
                        />
                        <label htmlFor="file-upload" className="file-label">
                            📁 {selectedFile ? selectedFile.name : 'Choose Audio File'}
                        </label>
                        <button
                            onClick={handleFileUpload}
                            disabled={!selectedFile || isLoading}
                            className="upload-button"
                        >
                            Upload & Search
                        </button>
                    </div>

                    <div className="test-section">
                        <button
                            onClick={handleRandomTest}
                            disabled={isLoading}
                            className="test-button"
                        >
                            Random Test
                        </button>
                    </div>
                </section>

                {isLoading && (
                    <div className="loading">
                        <div className="spinner"></div>
                        <p>Processing audio...</p>
                    </div>
                )}

                {error && (
                    <div className="error">
                        <p>❌ {error}</p>
                    </div>
                )}

                {results.length > 0 && (
                    <section className="results-section">
                        <h2>Results {testFile && <span className="test-file">(Test: {testFile})</span>}</h2>
                        <ul className="results-list">
                            {results.map((match, index) => (
                                <li key={index} className={`result-item ${index === 0 ? 'best-match' : ''}`}>
                                    <span className="rank">#{match.rank}</span>
                                    <div className="song-info">
                                        <span className="title">{match.title}</span>
                                        <span className="artist">{match.artist}</span>
                                    </div>
                                    <div className="scores">
                                        <span className="dtw-score">DTW: {match.dtw_score}</span>
                                        <span className="similarity">{match.similarity}%</span>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </section>
                )}

                <section className="songs-section">
                    <h2>📚 Database ({songs.length} songs)</h2>
                    <ul className="songs-list">
                        {songs.map((song) => (
                            <li key={song.id}>
                                <strong>{song.title}</strong> — {song.artist}
                            </li>
                        ))}
                    </ul>
                </section>
            </main>
        </div>
    );
}

export default App;