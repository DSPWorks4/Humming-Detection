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

    // Default to TRUE for Dark Mode
    const [darkMode, setDarkMode] = useState(() => {
        const saved = localStorage.getItem('darkMode');
        return saved !== null ? saved === 'true' : true;
    });

    const fileInputRef = useRef(null);

    // Save dark mode preference & Apply Theme
    useEffect(() => {
        localStorage.setItem('darkMode', darkMode);
        // We set the attribute on the body for global styling
        document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    }, [darkMode]);

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
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <div className="header-content">
                    <div className="header-title">
                        <h1>🎵 Hum-to-Search</h1>
                        <p>Discover music with your voice</p>
                    </div>
                    <button
                        className="theme-toggle"
                        onClick={() => setDarkMode(!darkMode)}
                        aria-label="Toggle Theme"
                    >
                        {darkMode ? '☀️' : '🌙'}
                    </button>
                </div>
            </header>

            <main>
                <section className="recorder-section">
                    <Recorder onRecordingComplete={handleRecordingComplete} />

                    <div className="divider">
                        <span>or upload file</span>
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
                            {selectedFile ? (
                                <>📄 {selectedFile.name}</>
                            ) : (
                                <>📁 Select Audio File</>
                            )}
                        </label>
                        <button
                            onClick={handleFileUpload}
                            disabled={!selectedFile || isLoading}
                            className="upload-button"
                        >
                            Analyze File
                        </button>

                        <button
                            onClick={handleRandomTest}
                            disabled={isLoading}
                            className="test-button"
                        >
                            🎲 Run Random Test
                        </button>
                    </div>
                </section>

                {isLoading && (
                    <div className="loading">
                        <div className="spinner"></div>
                        <p>Listening & Analyzing...</p>
                    </div>
                )}

                {error && (
                    <div className="error">
                        <p>{error}</p>
                    </div>
                )}

                {results.length > 0 && (
                    <section className="results-section">
                        <h2>
                            Match Results
                            {testFile && <span style={{ fontSize: '0.8rem', opacity: 0.7, marginLeft: '10px' }}> (Test: {testFile})</span>}
                        </h2>
                        <ul className="results-list">
                            {results.map((match, index) => (
                                <li key={index} className={`result-item ${index === 0 ? 'best-match' : ''}`}>
                                    <span className="rank">{index + 1}</span>
                                    <div className="song-info">
                                        <span className="title">{match.title}</span>
                                        <span className="artist">{match.artist}</span>
                                    </div>
                                    <div className="scores">
                                        <span className="similarity">{match.similarity}%</span>
                                        <span className="dtw-score">Distance: {match.dtw_score}</span>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </section>
                )}

                <section className="songs-section">
                    <h2>📚 Database Index ({songs.length})</h2>
                    <ul className="songs-list">
                        {songs.map((song) => (
                            <li key={song.id}>
                                <strong>{song.title}</strong> · {song.artist}
                            </li>
                        ))}
                    </ul>
                </section>
            </main>
        </div>
    );
}

export default App;