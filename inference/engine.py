from TTS.utils.synthesizer import Synthesizer

# Initialize Synthesizer
synth = Synthesizer(
    tts_checkpoint="models/best_model.pth",
    tts_config_path="models/config.json",
    vocoder_checkpoint="models/hifigan_model.pth",
    vocoder_config="models/hifigan_config.json",
    speaker_wav="audio_samples/viraj.wav",  # Default speaker
    use_cuda=False
)

def generate_voice(text: str, output_path="audio_samples/output.mp3"):
    synth.tts_to_file(text=text, file_path=output_path)
    return output_path