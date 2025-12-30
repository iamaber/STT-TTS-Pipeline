# Frontend

Simple HTML/CSS/JS interface for testing the STT-TTS pipeline.

## Features

- **Speech to Text**: Record audio from microphone and get real-time transcription
- **Text to Speech**: Enter text and generate speech with speaker selection (12,800 speakers)
- **Full Pipeline**: Upload audio file for STT→TTS processing

## Access

Open in browser: http://localhost:8000/frontend/index.html

## Usage

1. Make sure the server is running
2. Open the frontend URL in your browser
3. Allow microphone access when prompted
4. Test the different features:
   - Click 'Start Recording' to record audio
   - Enter text and click 'Synthesize' for TTS
   - Upload an audio file for full pipeline test

## Files

- `index.html` - Main HTML structure
- `style.css` - Modern gradient styling
- `app.js` - API integration and microphone handling
