const API_URL = window.location.origin;

// Configuration loaded from backend
let CONFIG = {
    asr: { sample_rate: 16000, chunk_size_ms: 100 },
    tts: { sample_rate: 44100, default_speaker_id: 50 },  // Will be updated from backend
    streaming: { sample_rate: 16000, max_buffer_seconds: 60 },
    llm_enabled: false
};

// Audio processing constants
const AUDIO_CHUNK_SIZE = 4096; // Size of audio chunks for streaming
const POLL_INTERVAL = 500; // Interval for polling audio queue (ms)

// GLOBAL STATE
let mediaRecorder = null;
let audioContext = null;
let audioWorkletNode = null;
let sessionId = null;   
let isRecording = false;
let isASRMuted = false;  // Track ASR mute state
let currentAudioQueue = [];
let isPlayingAudio = false;
let isPolling = false;
let currentAudioSource = null;  // Track current audio source for interruption
let audioPlaybackContext = null;  // Reuse AudioContext for playback

// ============================================
// DOM ELEMENTS
// ============================================
const elements = {
    // Buttons
    startBtn: document.getElementById('start-btn'),
    stopBtn: document.getElementById('stop-btn'),
    resetBtn: document.getElementById('reset-btn'),
    speakerSelect: document.getElementById('speaker-select'),
    
    // Status indicators
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    queueStatus: document.getElementById('queue-status'),
    
    // ASR (User) section
    asrStatus: document.getElementById('asr-status'),
    asrTyping: document.getElementById('asr-typing'),
    asrText: document.getElementById('asr-text'),
    
    // LLM (Assistant) section
    llmStatus: document.getElementById('llm-status'),
    llmText: document.getElementById('llm-text'),
    llmCursor: document.getElementById('llm-cursor'),
    
    // TTS section
    ttsText: document.getElementById('tts-text'),
    ttsProgress: document.getElementById('tts-progress'),
    queueCount: document.getElementById('queue-count'),
    
    // History
    conversationHistory: document.getElementById('conversation-history')
};

// ============================================
// CONVERSATION UI CLASS
// Manages all UI updates for real-time streaming
// ============================================
class ConversationUI {
    constructor() {
        this.currentLLMText = '';
        this.currentSentenceIndex = 0;
        this.totalSentences = 0;
    }
    
    // ----------------------------------------
    // ASR (User Speech) Updates
    // ----------------------------------------
    updateASR(text, isActive = true) {
        // Format text to show line-by-line (replace timestamps with newlines)
        const formattedText = text.replace(/\]\s*/g, ']\n') || 'Speak to start...';
        elements.asrText.textContent = formattedText;
        
        if (isActive) {
            elements.asrTyping.classList.add('active');
            elements.asrStatus.textContent = 'Speaking...';
            elements.asrStatus.classList.add('active');
        } else {
            elements.asrTyping.classList.remove('active');
            elements.asrStatus.textContent = 'Completed';
            elements.asrStatus.classList.remove('active');
        }
    }
    
    // ----------------------------------------
    // LLM Streaming Updates
    // ----------------------------------------
    startLLMStreaming() {
        this.currentLLMText = '';
        elements.llmText.textContent = '';
        elements.llmCursor.classList.add('active');
        elements.llmStatus.textContent = 'Thinking...';
        elements.llmStatus.classList.add('active');
    }
    
    appendLLMText(text) {
        // Append with a guaranteed single separating space to avoid stuck words
        const chunk = text.trim();
        if (!chunk) return;
        if (this.currentLLMText && !this.currentLLMText.endsWith(' ')) {
            this.currentLLMText += ' ';
        }
        this.currentLLMText += chunk + ' ';
        elements.llmText.textContent = this.currentLLMText;
        // Auto-scroll to show latest text
        elements.llmText.scrollTop = elements.llmText.scrollHeight;
    }
    
    completeLLMSentence() {
        this.currentLLMText += ' ';
        elements.llmText.textContent = this.currentLLMText;
    }
    
    stopLLMStreaming() {
        elements.llmCursor.classList.remove('active');
        elements.llmStatus.textContent = 'Complete';
        elements.llmStatus.classList.remove('active');
    }
    
    // ----------------------------------------
    // TTS Playback Updates
    // ----------------------------------------
    updateTTSStatus(text, progress = 0, queueSize = 0) {
        elements.ttsText.textContent = text;
        elements.ttsProgress.style.width = `${progress}%`;
        elements.queueCount.textContent = `Queue: ${queueSize} sentence${queueSize !== 1 ? 's' : ''}`;
    }
    
    startTTSPlayback(sentenceText, sentenceIndex, totalSentences) {
        this.currentSentenceIndex = sentenceIndex;
        this.totalSentences = totalSentences;
        
        const displayText = sentenceText.length > 50 
            ? `Speaking: "${sentenceText.substring(0, 50)}..."` 
            : `Speaking: "${sentenceText}"`;
            
        this.updateTTSStatus(
            displayText,
            0,
            totalSentences - sentenceIndex - 1
        );
    }
    
    updateTTSProgress(progress) {
        this.updateTTSStatus(
            elements.ttsText.textContent,
            progress,
            this.totalSentences - this.currentSentenceIndex - 1
        );
    }
    
    completeTTSPlayback() {
        this.updateTTSStatus('Ready to speak', 0, 0);
    }
    
    // ----------------------------------------
    // Overall Status Updates
    // ----------------------------------------
    updateStatus(statusClass, text, queueInfo = null) {
        elements.statusDot.className = `status-dot ${statusClass}`;
        elements.statusText.textContent = text;
        
        if (queueInfo) {
            elements.queueStatus.innerHTML = `<small>${queueInfo}</small>`;
        } else {
            elements.queueStatus.innerHTML = '<small>Queue: 0 messages</small>';
        }
    }
    
    // ----------------------------------------
    // Conversation History
    // ----------------------------------------
    addToHistory(role, text) {
        // Remove placeholder if exists
        const placeholder = elements.conversationHistory.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        // Create history item
        const item = document.createElement('div');
        item.className = `history-item ${role}`;
        item.innerHTML = `
            <div class="role">${role === 'user' ? '👤 You' : '🤖 Assistant'}</div>
            <div>${text}</div>
        `;
        
        elements.conversationHistory.appendChild(item);
        // Auto-scroll to bottom
        elements.conversationHistory.scrollTop = elements.conversationHistory.scrollHeight;
    }
}

// Initialize UI manager
const conversationUI = new ConversationUI();

// ============================================
// AUDIO RECORDING & STREAMING
// ============================================
async function startRecording() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Initialize audio context with config sample rate
        audioContext = new AudioContext({ sampleRate: CONFIG.asr.sample_rate });
        const source = audioContext.createMediaStreamSource(stream);
        
        // Create audio processor
        await audioContext.audioWorklet.addModule(createAudioProcessorCode());
        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
        
        // Connect audio pipeline
        source.connect(audioWorkletNode);
        audioWorkletNode.connect(audioContext.destination);
        
        // Handle audio chunks
        audioWorkletNode.port.onmessage = async (event) => {
            // Only send audio if recording is active AND not muted
            if (isRecording && !isASRMuted) {
                await sendAudioChunk(event.data);
            }
        };
        
        // Generate session ID
        sessionId = `session_${Date.now()}`;
        isRecording = true;
        isASRMuted = false;  // Reset mute state
        
        // Update UI
        elements.startBtn.disabled = true;
        elements.stopBtn.disabled = false;
        elements.stopBtn.innerHTML = '<span class="icon">🔇</span> Mute';
        elements.stopBtn.classList.remove('btn-warning');
        elements.stopBtn.classList.add('btn-danger');
        conversationUI.updateStatus('recording', 'Recording', null);
        conversationUI.updateASR('Listening...', true);
        
        console.log('Recording started');
    } catch (error) {
        console.error('Failed to start recording:', error);
        alert('Microphone access denied or not available');
    }
}

function toggleASRMute() {
    if (!isRecording) return;  // Only works when recording is active
    
    isASRMuted = !isASRMuted;
    
    if (isASRMuted) {
        // Mute ASR - stop sending audio to backend (LLM/TTS continue normally)
        
        // Update UI to show muted state
        elements.stopBtn.innerHTML = '<span class="icon">🔊</span> Unmute';
        elements.stopBtn.classList.remove('btn-danger');
        elements.stopBtn.classList.add('btn-warning');
        conversationUI.updateStatus('muted', 'ASR Muted (Mic Off)', null);
        conversationUI.updateASR('🔇 Microphone muted - LLM & TTS still active', false);
        console.log('ASR muted - microphone input disabled');
    } else {
        // Unmute ASR - resume sending audio to backend
        
        // Update UI to show active state
        elements.stopBtn.innerHTML = '<span class="icon">🔇</span> Mute';
        elements.stopBtn.classList.remove('btn-warning');
        elements.stopBtn.classList.add('btn-danger');
        conversationUI.updateStatus('recording', 'Recording', null);
        conversationUI.updateASR('Listening...', true);
        console.log('ASR unmuted - microphone input enabled');
    }
}


function stopAudio() {
    // Stop currently playing audio
    if (currentAudioSource) {
        try {
            currentAudioSource.stop();
            currentAudioSource = null;
        } catch (e) {
            // Already stopped
        }
    }
    
    // Clear audio queue
    currentAudioQueue = [];
    isPlayingAudio = false;
    
    console.log('Audio playback stopped');
}

async function resetSession() {
    // Stop recording if active
    if (isRecording) {
        isRecording = false;
        isASRMuted = false;
        
        // Cleanup audio resources
        if (audioWorkletNode) {
            audioWorkletNode.disconnect();
            audioWorkletNode = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }
    stopAllAudioPlayback();
    
    // Stop audio playback
    isPolling = false;
    currentAudioQueue = [];
    isPlayingAudio = false;
    
    // Reset UI
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.stopBtn.innerHTML = '<span class="icon">🔇</span> Mute';
    elements.stopBtn.classList.remove('btn-warning');
    elements.stopBtn.classList.add('btn-danger');
    
    // Reset session on server
    if (sessionId) {
        try {
            await fetch(`${API_URL}/api/stream/reset?session_id=${sessionId}`, {
                method: 'POST'
            });
        } catch (error) {
            console.error('Reset error:', error);
        }
    }
    
    // Clear external LLM memory
    try {
        await fetch('http://192.168.10.2:8000/api/memory/clear', {
            method: 'POST',
            headers: { 'accept': 'application/json' }
        });
        console.log('External LLM memory cleared');
    } catch (error) {
        console.error('Failed to clear external LLM memory:', error);
    }
    
    // Reset UI
    sessionId = null;
    window.lastSentTranscript = null;  // Reset transcript tracker
    conversationUI.currentLLMText = '';
    elements.asrText.textContent = 'Speak to start...';
    elements.llmText.textContent = 'Waiting for your message...';
    elements.conversationHistory.innerHTML = '<p class="placeholder">Conversation history will appear here...</p>';
    conversationUI.updateStatus('ready', 'Ready', null);
    conversationUI.completeTTSPlayback();
    
    console.log('Session reset');
}

// ============================================
// AUDIO PROCESSING
// ============================================
function createAudioProcessorCode() {
    // Create inline audio processor worklet
    const processorCode = `
        class AudioProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                this.bufferSize = ${AUDIO_CHUNK_SIZE};
                this.buffer = new Float32Array(this.bufferSize);
                this.bufferIndex = 0;
            }
            
            process(inputs) {
                const input = inputs[0];
                if (input.length > 0) {
                    const channelData = input[0];
                    
                    for (let i = 0; i < channelData.length; i++) {
                        this.buffer[this.bufferIndex++] = channelData[i];
                        
                        if (this.bufferIndex >= this.bufferSize) {
                            this.port.postMessage(this.buffer.slice());
                            this.bufferIndex = 0;
                        }
                    }
                }
                return true;
            }
        }
        
        registerProcessor('audio-processor', AudioProcessor);
    `;
    
    const blob = new Blob([processorCode], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
}

async function sendAudioChunk(audioData) {
    try {
        // Convert float32 to int16
        const int16Data = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
            int16Data[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
        }
        
        // Encode to base64
        const base64Audio = btoa(String.fromCharCode(...new Uint8Array(int16Data.buffer)));
        
        // Send to server (backend will use default speaker from config)
        const response = await fetch(`${API_URL}/api/stream/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                audio: base64Audio,
                sample_rate: CONFIG.asr.sample_rate
                // speaker_id removed - backend uses default from config
            })
        });
        
        const result = await response.json();
        
        // Update ASR transcription (show all accumulated transcripts)
        if (result.transcription) {
            conversationUI.updateASR(result.transcription, true);
            
            // Only stop playback if this is NEW speech (different from last sent to LLM)
            const lines = result.transcription.trim().split('\n');
            const latestLine = lines[lines.length - 1];
            const latestTranscript = latestLine.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, '').trim();
            
            // Stop audio only if we detect genuinely new speech
            if (latestTranscript && window.lastSentTranscript && latestTranscript !== window.lastSentTranscript && isPlayingAudio) {
                console.log('[INTERRUPT] New speech detected, stopping playback');
                stopAllAudioPlayback();
            }
        }
        
        // If TTS audio is generated, extract the LATEST transcript and send to LLM
        // The transcription contains all accumulated transcripts with timestamps
        // Format: "[14:08:36] hello\n[14:08:45] mohi moshi"
        if (result.audio && result.transcription) {
            // Extract only the last line (latest transcript)
            const lines = result.transcription.trim().split('\n');
            const latestLine = lines[lines.length - 1];
            
            // Remove timestamp: "[14:08:36] hello" -> "hello"
            const latestTranscript = latestLine.replace(/^\[\d{2}:\d{2}:\d{2}\]\s*/, '').trim();
            
            // Only send if this is a new, different transcription
            if (latestTranscript && (!window.lastSentTranscript || window.lastSentTranscript !== latestTranscript)) {
                console.log(`[LLM] Sending to LLM: "${latestTranscript}"`);
                window.lastSentTranscript = latestTranscript;
                await sendToLLM(latestTranscript);
            } else if (latestTranscript) {
                console.log(`[LLM] Skipping duplicate: "${latestTranscript}"`);
            }
        }
        
    } catch (error) {
        console.error('Audio chunk error:', error);
    }
}

// ============================================
// LLM CONVERSATION
// ============================================
async function sendToLLM(transcription) {
    // Interrupt if user types a stop command
    if (isStopCommand(transcription)) {
        stopAllAudioPlayback();
        // Optionally, clear backend TTS queue (if endpoint exists)
        // await fetch(`${API_URL}/api/conversation/audio/cleanup`, { method: 'POST' });
        conversationUI.stopLLMStreaming();
        conversationUI.completeTTSPlayback();
        conversationUI.updateStatus('ready', 'Ready', null);
        conversationUI.addToHistory('assistant', 'Stopped talking as requested.');
        return;
    }
    
    try {
        // Add to history
        conversationUI.addToHistory('user', transcription);
        conversationUI.updateASR(transcription, false);
        
        // Start LLM streaming UI
        conversationUI.startLLMStreaming();
        conversationUI.updateStatus('processing', 'Thinking', null);
        
        // Get speaker ID from input
        const speakerInput = document.getElementById('speaker-select');
        const speakerId = speakerInput ? parseInt(speakerInput.value) : CONFIG.tts.default_speaker_id;
        
        // Send to LLM with speaker ID
        const response = await fetch(`${API_URL}/api/conversation/send`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: transcription,
                session_id: sessionId,
                speaker_id: speakerId  // Use speaker ID from input
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'processing') {
            // Start polling for audio
            if (!isPolling) {
                pollForAudio();
            }
        } else if (result.status === 'queued') {
            conversationUI.updateStatus('processing', 'Queued', `Position ${result.position} in queue`);
        }
        
    } catch (error) {
        console.error('LLM error:', error);
        conversationUI.stopLLMStreaming();
    }
}

// ============================================
// AUDIO PLAYBACK QUEUE
// ============================================
async function pollForAudio() {
    isPolling = true;
    let emptyPollCount = 0;
    const MAX_EMPTY_POLLS = 20; // Wait up to 10 seconds (20 * 500ms) before giving up
    
    console.log('[Polling] Started polling for audio');
    
    while (isPolling && emptyPollCount < MAX_EMPTY_POLLS) {
        try {
            const response = await fetch(`${API_URL}/api/conversation/audio/next`);
            const result = await response.json();
            
            if (result.audio_id && result.audio) {
                // Reset empty poll counter when we get audio
                emptyPollCount = 0;
                
                // Add to audio queue
                currentAudioQueue.push(result);
                console.log(`[Polling] Received audio: "${result.text}"`);
                
                // Add to conversation history
                if (result.text) {
                    conversationUI.appendLLMText(result.text + ' ');
                    conversationUI.addToHistory('assistant', result.text);
                }
                
                // Start playback if not already playing
                if (!isPlayingAudio) {
                    playNextAudio();
                }
            } else {
                // No audio available, increment counter
                emptyPollCount++;
                
                // If we have audio in queue, reset counter (still processing)
                if (currentAudioQueue.length > 0) {
                    emptyPollCount = 0;
                }
            }
            
            // Wait before next poll
            await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
            
        } catch (error) {
            console.error('[Polling] Error:', error);
            await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
        }
    }
    
    // Stop polling
    isPolling = false;
    console.log('[Polling] Stopped polling');
    
    // Only stop LLM streaming if we actually timed out with no audio
    if (emptyPollCount >= MAX_EMPTY_POLLS && currentAudioQueue.length === 0) {
        conversationUI.stopLLMStreaming();
        conversationUI.updateStatus('ready', 'Ready', null);
    }
}

async function playNextAudio() {
    if (currentAudioQueue.length === 0) {
        isPlayingAudio = false;
        conversationUI.completeTTSPlayback();
        return;
    }
    
    isPlayingAudio = true;
    const audioItem = currentAudioQueue.shift();
    
    try {
        // Decode base64 audio properly
        console.log(`[DECODE START] b64 length: ${audioItem.audio.length} chars`);
        const binaryString = atob(audioItem.audio);
        console.log(`[DECODE] Binary string: ${binaryString.length} chars`);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        console.log(`[DECODE] Bytes: ${bytes.length} bytes`);
        
        // Create Int16Array from byte buffer
        const int16Array = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
        console.log(`[DECODE] Int16 samples: ${int16Array.length}`);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }
        
        const expectedDuration = float32Array.length / audioItem.sample_rate;
        console.log(`[PLAYBACK] "${audioItem.text.substring(0, 60)}..." -> ${float32Array.length} samples, ${audioItem.sample_rate}Hz, ${expectedDuration.toFixed(2)}s`);
        
        // Create or reuse AudioContext
        if (!audioPlaybackContext || audioPlaybackContext.state === 'closed') {
            audioPlaybackContext = new AudioContext();
        }
        const audioCtx = audioPlaybackContext;
        const audioBuffer = audioCtx.createBuffer(
            1,  // mono
            float32Array.length,
            audioItem.sample_rate  // Use the actual TTS sample rate (44100)
        );
        audioBuffer.getChannelData(0).set(float32Array);
        
        // Play audio
        const source = audioCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioCtx.destination);
        
        // Update UI
        conversationUI.startTTSPlayback(audioItem.text, 0, currentAudioQueue.length + 1);
        conversationUI.updateStatus('processing', 'Speaking', null);
        
        // Track playback progress
        const duration = audioBuffer.duration;
        const startTime = audioCtx.currentTime;
        const progressInterval = setInterval(() => {
            const elapsed = audioCtx.currentTime - startTime;
            const progress = Math.min(100, (elapsed / duration) * 100);
            conversationUI.updateTTSProgress(progress);
            
            if (progress >= 100) {
                clearInterval(progressInterval);
            }
        }, 100);
        
        // Handle playback end
        source.onended = async () => {
            clearInterval(progressInterval);
            
            // Clear reference
            if (currentAudioSource === source) {
                currentAudioSource = null;
            }
            
            // Cleanup audio file on server
            await fetch(`${API_URL}/api/conversation/audio/cleanup?audio_id=${audioItem.audio_id}`, {
                method: 'POST'
            });
            
            // Play next audio
            playNextAudio();
        };
        
        // Store reference for interruption
        currentAudioSource = source;
        source.start(0);
        
    } catch (error) {
        console.error('Playback error:', error);
        // Continue to next audio on error
        playNextAudio();
    }
}

// ============================================
// EVENT LISTENERS
// ============================================
elements.startBtn.addEventListener('click', startRecording);
elements.stopBtn.addEventListener('click', toggleASRMute);
elements.resetBtn.addEventListener('click', resetSession);

// ============================================
// INITIALIZATION
// ============================================
async function loadConfig() {
    try {
        const response = await fetch(`${API_URL}/api/config`);
        const config = await response.json();
        
        // Update global CONFIG
        CONFIG = config;
        
        // Update speaker ID input to show backend value
        const speakerInput = document.getElementById('speaker-select');
        if (speakerInput) {
            speakerInput.value = config.tts.default_speaker_id;
        }
        
        console.log('Config loaded:', CONFIG);
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

// Load config and initialize
loadConfig().then(() => {
    console.log('AI Voice Assistant initialized');
    conversationUI.updateStatus('ready', 'Ready', null);
});

// Utility: Stop all audio playback immediately
function stopAllAudioPlayback() {
    // Stop currently playing audio
    if (currentAudioSource) {
        try {
            currentAudioSource.stop();
            currentAudioSource = null;
        } catch (e) {
            // Already stopped
        }
    }
    // Clear audio queue and state
    currentAudioQueue = [];
    isPlayingAudio = false;
    if (audioPlaybackContext && audioPlaybackContext.state !== 'closed') {
        audioPlaybackContext.close();
        audioPlaybackContext = null;
    }
    console.log('All audio playback forcibly stopped');
}

// Utility: Check if a string is a stop command
function isStopCommand(text) {
    const stopWords = ['stop', 'cancel', 'be quiet', 'shut up', 'halt', 'enough'];
    const lower = text.trim().toLowerCase();
    return stopWords.some(word => lower.includes(word));
}

// Patch: Call stopAllAudioPlayback on interruption
// 1. On new speech detected in sendAudioChunk
if (latestTranscript && window.lastSentTranscript && latestTranscript !== window.lastSentTranscript && isPlayingAudio) {
    console.log('[INTERRUPT] New speech detected, stopping playback');
    stopAllAudioPlayback();
}
