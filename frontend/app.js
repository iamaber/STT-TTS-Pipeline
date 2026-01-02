// Configuration
const API_URL = 'http://localhost:8000';
const SAMPLE_RATE = 16000;

// State
let mediaRecorder = null;
let audioContext = null;
let sessionId = generateSessionId();
let isRecording = false;

// DOM Elements
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const resetBtn = document.getElementById('reset-btn');
const speakerSelect = document.getElementById('speaker-select');
const transcriptionBox = document.getElementById('transcription-box');
const audioPlayer = document.getElementById('audio-player');
const audioStatus = document.getElementById('audio-status');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// Event Listeners
startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
resetBtn.addEventListener('click', resetSession);

// Generate unique session ID
function generateSessionId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Update status
function updateStatus(status, text) {
    statusDot.className = `status-dot ${status}`;
    statusText.textContent = text;
}

// Start recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: SAMPLE_RATE,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });

        audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);

        source.connect(processor);
        processor.connect(audioContext.destination);

        processor.onaudioprocess = async (e) => {
            if (!isRecording) return;

            const audioData = e.inputBuffer.getChannelData(0);
            await processAudioChunk(audioData);
        };

        isRecording = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        updateStatus('recording', 'Recording...');

    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Failed to access microphone. Please check permissions.');
    }
}

// Stop recording
function stopRecording() {
    isRecording = false;
    
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
    updateStatus('ready', 'Ready');
}

// Process audio chunk
async function processAudioChunk(audioData) {
    try {
        // Convert float32 to int16
        const int16Data = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            int16Data[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
        }

        // Convert to base64
        const audioB64 = btoa(String.fromCharCode(...new Uint8Array(int16Data.buffer)));

        // Send to backend
        const response = await fetch(`${API_URL}/api/stream/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                audio: audioB64,
                sample_rate: SAMPLE_RATE,
                speaker: parseInt(speakerSelect.value)
            })
        });

        if (response.ok) {
            const result = await response.json();
            
            // Update transcription
            if (result.transcription && result.transcription !== 'Listening...') {
                updateTranscription(result.transcription);
            }

            // Play TTS audio
            if (result.audio) {
                playTTSAudio(result.audio, result.sample_rate);
            }
        }
    } catch (error) {
        console.error('Error processing audio:', error);
    }
}

// Update transcription display
function updateTranscription(text) {
    // Remove placeholder if exists
    const placeholder = transcriptionBox.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    // Parse transcription lines
    const lines = text.split('\n').filter(line => line.trim());
    
    // Clear and rebuild
    transcriptionBox.innerHTML = '';
    lines.forEach(line => {
        const div = document.createElement('div');
        div.className = 'transcript-line';
        
        // Extract timestamp and text
        const match = line.match(/\[(.*?)\] (.*)/);
        if (match) {
            const timestamp = match[1];
            const content = match[2];
            div.innerHTML = `<span class="timestamp">[${timestamp}]</span>${content}`;
        } else {
            div.textContent = line;
        }
        
        transcriptionBox.appendChild(div);
    });

    // Auto-scroll to bottom
    transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
}

// Play TTS audio
function playTTSAudio(audioB64, sampleRate) {
    try {
        // Decode base64
        const binaryString = atob(audioB64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Convert to float32
        const float32Array = new Float32Array(bytes.buffer);

        // Create audio context
        const audioCtx = new AudioContext({ sampleRate: sampleRate });
        const audioBuffer = audioCtx.createBuffer(1, float32Array.length, sampleRate);
        audioBuffer.getChannelData(0).set(float32Array);

        // Create source and play
        const source = audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioCtx.destination);
        source.start(0);

        // Update audio player for visual feedback
        audioStatus.textContent = `Playing TTS audio (${(float32Array.length / sampleRate).toFixed(2)}s)`;
        updateStatus('processing', 'Playing TTS...');

        source.onended = () => {
            if (isRecording) {
                updateStatus('recording', 'Recording...');
            } else {
                updateStatus('ready', 'Ready');
            }
        };

    } catch (error) {
        console.error('Error playing TTS audio:', error);
        audioStatus.textContent = 'Error playing audio';
    }
}

// Reset session
async function resetSession() {
    try {
        await fetch(`${API_URL}/api/stream/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        // Generate new session ID
        sessionId = generateSessionId();

        // Clear transcription
        transcriptionBox.innerHTML = '<p class="placeholder">Transcriptions will appear here...</p>';
        
        // Reset audio
        audioPlayer.src = '';
        audioStatus.textContent = 'No audio generated yet';

        updateStatus('ready', 'Ready');

    } catch (error) {
        console.error('Error resetting session:', error);
    }
}

// Initialize
updateStatus('ready', 'Ready');
console.log('STT-TTS Frontend initialized');
