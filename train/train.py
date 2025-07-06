import os
from TTS.tts.datasets.preprocess import process_dataset
from TTS.config.shared_configs import BaseTTSConfig
from TTS.tts.models.fastspeech2 import FastSpeech2
from TTS.tts.trainers import FastSpeech2Trainer
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.vocoder.models.hifigan import Hifigan
from TTS.vocoder.trainers import HifiganTrainer
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.utils.audio import AudioProcessor

# ✅ Paths
dataset_path = "train/dataset/viraj/"
output_path = "models/"
config_path = "train/config/fs2_config.json"

# ✅ Step 1: Preprocess dataset
print("🔄 Preprocessing dataset...")
process_dataset(
    config_path=config_path,
    root_dir=dataset_path,
    extension="wav",
    num_workers=2
)

# ✅ Step 2: Train FastSpeech2
print("🚀 Training FastSpeech2...")
fs2_config = BaseTTSConfig()
fs2_config.output_path = output_path
fs2_model = FastSpeech2(config=fs2_config)
trainer = FastSpeech2Trainer(
    config=fs2_config,
    model=fs2_model,
    output_path=output_path
)
trainer.fit()

# ✅ Step 3: Train HiFi-GAN Vocoder
print("🎧 Training HiFi-GAN...")
hifi_config = HifiganConfig()
hifi_config.output_path = output_path
ap = AudioProcessor(**hifi_config.audio)
vocoder_model = Hifigan(config=hifi_config, ap=ap)
vocoder_trainer = HifiganTrainer(
    config=hifi_config,
    model=vocoder_model,
    output_path=output_path
)
vocoder_trainer.fit()

print("✅ Training complete. Models saved in /models/")