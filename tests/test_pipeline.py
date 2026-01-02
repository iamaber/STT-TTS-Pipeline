import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.audio import load_audio, save_audio
from app.services.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Test STT-TTS pipeline locally")
    parser.add_argument(
        "--audio", type=str, required=True, help="Input audio file path"
    )
    parser.add_argument(
        "--output", type=str, default="output.wav", help="Output audio file path"
    )
    parser.add_argument(
        "--speaker", type=int, default=92, help="Speaker ID for TTS (default: 92)"
    )

    args = parser.parse_args()

    print("Loading pipeline...")
    pipeline = Pipeline()

    print(f"Loading audio from {args.audio}...")
    audio, sr = load_audio(args.audio)
    print(f"Loaded {len(audio)} samples at {sr}Hz")

    print("Processing full pipeline (STT → TTS)...")
    transcription, output_audio = pipeline.process_full_pipeline(
        audio, sr, args.speaker
    )

    print(f'\nTranscription: "{transcription}"')

    if len(output_audio) > 0:
        print(f"Saving output to {args.output}...")
        save_audio(output_audio, args.output, pipeline.tts.tts.sample_rate)
        print(f"✅ Done! Generated {len(output_audio)} samples")
    else:
        print("⚠️  No speech detected in input audio.")


if __name__ == "__main__":
    main()
