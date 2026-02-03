import React, { useState, useRef, useEffect } from 'react';

const Recorder = ({ onRecordingComplete }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [audioURL, setAudioURL] = useState(null);

    const mediaRecorderRef = useRef(null);
    const chunksRef = useRef([]);
    const timerRef = useRef(null);
    const streamRef = useRef(null);

    const MAX_RECORDING_TIME = 30; // seconds

    useEffect(() => {
        return () => {
            // Cleanup on unmount
            if (timerRef.current) clearInterval(timerRef.current);
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100,
                }
            });

            streamRef.current = stream;
            chunksRef.current = [];

            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            mediaRecorderRef.current = mediaRecorder;

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
                const url = URL.createObjectURL(blob);
                setAudioURL(url);
                onRecordingComplete(blob);

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start(100); // Collect data every 100ms
            setIsRecording(true);
            setRecordingTime(0);
            setAudioURL(null);

            // Start timer
            timerRef.current = setInterval(() => {
                setRecordingTime(prev => {
                    if (prev >= MAX_RECORDING_TIME - 1) {
                        stopRecording();
                        return MAX_RECORDING_TIME;
                    }
                    return prev + 1;
                });
            }, 1000);

        } catch (err) {
            console.error('Error accessing microphone:', err);
            alert('Could not access microphone. Please check permissions.');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);

            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
        }
    };

    const formatTime = (seconds) => {
        return `${Math.floor(seconds / 60)}:${(seconds % 60).toString().padStart(2, '0')}`;
    };

    return (
        <div className="recorder">
            <div className="recorder-controls">
                {!isRecording ? (
                    <button onClick={startRecording} className="record-button">
                        🎤 Start Recording
                    </button>
                ) : (
                    <button onClick={stopRecording} className="stop-button">
                        ⏹️ Stop Recording
                    </button>
                )}
            </div>

            {isRecording && (
                <div className="recording-indicator">
                    <div className="pulse"></div>
                    <span>Recording: {formatTime(recordingTime)} / {formatTime(MAX_RECORDING_TIME)}</span>
                </div>
            )}

            {audioURL && (
                <div className="playback">
                    <audio controls src={audioURL} />
                </div>
            )}

            <p className="recorder-hint">
                Hum for 10-30 seconds for best results
            </p>
        </div>
    );
};

export default Recorder;