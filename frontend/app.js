const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';

const synthesizeBtn = document.getElementById('synthesizeBtn');
const textInput = document.getElementById('textInput');
const speakerSelect = document.getElementById('speakerSelect');
const audioPlayer = document.getElementById('audioPlayer');
const pipelineBtn = document.getElementById('pipelineBtn');
const audioFile = document.getElementById('audioFile');
const pipelineTranscription = document.getElementById('pipelineTranscription');
const pipelineAudio = document.getElementById('pipelineAudio');
const statusDiv = document.getElementById('status');
const streamStartBtn = document.getElementById('streamStartBtn');
const streamStopBtn = document.getElementById('streamStopBtn');
const streamTranscription = document.getElementById('streamTranscription');
const streamStatus = document.getElementById('streamStatus');

let streamingWs;
let streamingMediaRecorder;
let audioContext;

function setStatus(message, type = 'ready') {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
}

synthesizeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (!text) {
        setStatus('Please enter text to synthesize', 'error');
        return;
    }

    try {
        setStatus('Synthesizing...', 'processing');
        const speaker = speakerSelect.value;
        const response = await fetch(`${API_URL}/synthesize?text=${encodeURIComponent(text)}&speaker=${speaker}`, {
            method: 'POST'
        });

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        audioPlayer.src = audioUrl;
        audioPlayer.play();
        setStatus('Synthesis complete', 'success');
    } catch (error) {
        setStatus('Synthesis failed: ' + error.message, 'error');
    }
});

pipelineBtn.addEventListener('click', async () => {
    const file = audioFile.files[0];
    if (!file) {
        setStatus('Please select an audio file', 'error');
        return;
    }

    try {
        setStatus('Processing pipeline...', 'processing');
        const formData = new FormData();
        formData.append('file', file);
        const speaker = speakerSelect.value;

        const response = await fetch(`${API_URL}/pipeline?speaker=${speaker}`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        pipelineTranscription.textContent = data.transcription || 'No speech detected';
        setStatus('Pipeline complete', 'success');
    } catch (error) {
        setStatus('Pipeline failed: ' + error.message, 'error');
    }
});

streamStartBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            } 
        });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        streamingWs = new WebSocket(`${WS_URL}/ws/stream`);
        
        streamingWs.onopen = () => {
            streamStatus.textContent = 'Streaming active - Speak now!';
            streamStatus.className = 'stream-status active';
            streamStartBtn.disabled = true;
            streamStopBtn.disabled = false;
            setStatus('Streaming...', 'recording');
        };
        
        streamingWs.onmessage = async (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'transcription') {
                const timestamp = new Date().toLocaleTimeString();
                const transcriptDiv = document.createElement('div');
                transcriptDiv.style.padding = '8px';
                transcriptDiv.style.marginBottom = '5px';
                transcriptDiv.style.background = '#e8f5e9';
                transcriptDiv.style.borderRadius = '4px';
                transcriptDiv.innerHTML = `<strong>[${timestamp}]</strong> ${data.text}`;
                streamTranscription.appendChild(transcriptDiv);
                streamTranscription.scrollTop = streamTranscription.scrollHeight;
            } else if (data.type === 'audio') {
                const audioData = atob(data.audio);
                const audioArray = new Float32Array(audioData.length / 4);
                const dataView = new DataView(new ArrayBuffer(audioData.length));
                
                for (let i = 0; i < audioData.length; i++) {
                    dataView.setUint8(i, audioData.charCodeAt(i));
                }
                
                for (let i = 0; i < audioArray.length; i++) {
                    audioArray[i] = dataView.getFloat32(i * 4, true);
                }
                
                playAudioChunk(audioArray, data.sample_rate);
            }
        };
        
        streamingWs.onerror = (error) => {
            console.error('WebSocket error:', error);
            setStatus('WebSocket error', 'error');
        };
        
        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        processor.onaudioprocess = (e) => {
            if (streamingWs && streamingWs.readyState === WebSocket.OPEN) {
                const float32Audio = e.inputBuffer.getChannelData(0);
                
                const int16Audio = new Int16Array(float32Audio.length);
                for (let i = 0; i < float32Audio.length; i++) {
                    const s = Math.max(-1, Math.min(1, float32Audio[i]));
                    int16Audio[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                const audioB64 = btoa(String.fromCharCode(...new Uint8Array(int16Audio.buffer)));
                streamingWs.send(JSON.stringify({
                    type: 'audio',
                    audio: audioB64
                }));
            }
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        streamingMediaRecorder = { stream, source, processor };
        
    } catch (error) {
        console.error('Streaming error:', error);
        setStatus('Streaming error: ' + error.message, 'error');
    }
});

streamStopBtn.addEventListener('click', () => {
    if (streamingWs) {
        streamingWs.close();
    }
    if (streamingMediaRecorder) {
        streamingMediaRecorder.processor.disconnect();
        streamingMediaRecorder.source.disconnect();
        streamingMediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }
    
    streamStatus.textContent = 'Not streaming';
    streamStatus.className = 'stream-status';
    streamStartBtn.disabled = false;
    streamStopBtn.disabled = true;
    setStatus('Streaming stopped', 'ready');
});

async function playAudioChunk(audioData, sampleRate) {
    const playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
    const buffer = playbackContext.createBuffer(1, audioData.length, sampleRate);
    buffer.getChannelData(0).set(audioData);
    
    const source = playbackContext.createBufferSource();
    source.buffer = buffer;
    source.connect(playbackContext.destination);
    source.start();
}

setStatus('Ready');
