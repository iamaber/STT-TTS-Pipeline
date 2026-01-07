from pathlib import Path
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.collections.tts.models import FastPitchModel, HifiGanModel


def download_models():
    """Download all required NeMo models for the STT-TTS pipeline"""
    models_dir = Path("models")
    asr_dir = models_dir / "asr"
    tts_acoustic_dir = models_dir / "tts_acoustic"
    tts_vocoder_dir = models_dir / "tts_vocoder"

    # Create directories
    for directory in [asr_dir, tts_acoustic_dir, tts_vocoder_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading NeMo Models")
    print("=" * 60)
    print()

    # Download ASR model - CTC model (GPU stable)
    asr_path = asr_dir / "parakeet-ctc-1.1b.nemo"
    if asr_path.exists():
        print("1/3 Parakeet-CTC-1.1B ASR model already exists")
        print(f"    ✓ Found at {asr_path}")
    else:
        print("1/3 Downloading Parakeet-CTC-1.1B ASR model...")
        print("    Model: nvidia/parakeet-ctc-1.1b")
        try:
            asr_model = EncDecCTCModelBPE.from_pretrained(
                "nvidia/parakeet-ctc-1.1b"
            )
            asr_model.save_to(str(asr_path))
            print(f"    ✓ Saved to {asr_path}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            return False
    print()

    # Download TTS acoustic model
    fp_path = tts_acoustic_dir / "tts_en_fastpitch_multispeaker.nemo"
    if fp_path.exists():
        print("2/3 FastPitch TTS acoustic model already exists")
        print(f"    ✓ Found at {fp_path}")
    else:
        print("2/3 Downloading FastPitch TTS acoustic model...")
        print("    Model: Mastering-Python-HF/nvidia_tts_en_fastpitch_multispeaker")

        try:
            fastpitch = FastPitchModel.from_pretrained(
                "Mastering-Python-HF/nvidia_tts_en_fastpitch_multispeaker"
            )
            fastpitch.save_to(str(fp_path))
            print(f"    ✓ Saved to {fp_path}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            print(
                "    Note: If this fails, the model may already be in models/tts_acoustic/"
            )
            if not fp_path.exists():
                return False
    print()

    # Download TTS vocoder
    hg_path = tts_vocoder_dir / "tts_en_hifitts_hifigan_ft_fastpitch.nemo"
    if hg_path.exists():
        print("3/3 HiFi-GAN vocoder already exists")
        print(f"    ✓ Found at {hg_path}")
    else:
        print("3/3 Downloading HiFi-GAN vocoder...")
        print("    Model: nvidia/tts_en_hifitts_hifigan_ft_fastpitch")

        try:
            hifigan = HifiGanModel.from_pretrained(
                "nvidia/tts_en_hifitts_hifigan_ft_fastpitch"
            )
            hifigan.save_to(str(hg_path))
            print(f"    ✓ Saved to {hg_path}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            print(
                "    Note: If this fails, the model may already be in models/tts_vocoder/"
            )
            if not hg_path.exists():
                return False
    print()

    print("✅ All models ready!")
    return True


if __name__ == "__main__":
    import sys

    success = download_models()
    sys.exit(0 if success else 1)
