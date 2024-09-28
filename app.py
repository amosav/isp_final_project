import sys

import librosa
import torch
import torchaudio
from torch.nn.functional import cosine_similarity
from transformers import AutoProcessor

from models.model_utils import get_CLAP_LoRa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SAMPLING_RATE = 48000

CHECKPOINT_PATH = r"C:\Users\amos1\Downloads\music_genres_model_epoch_12_lr0.001_a4r4_noise.pt"

def get_model(path, r, alpha):
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    model = get_CLAP_LoRa(r=4, alpha=4)
    model.to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model, processor


def main():
    print("Select model type (music_genres or esc50):")
    model_type = input().strip()

    if model_type == "music_genres":
        path = r"C:\Users\amos1\Downloads\music_genres_model_epoch_12_lr0.001_a4r4_noise.pt"
    elif model_type == "esc50":
        path = r"C:\Users\amos1\Downloads\music_genres_model_epoch_12_lr0.001_a4r4.pt"
    else:
        print("Invalid model type. Exiting.")
        return

    model, processor = get_model(path, 4, 4)
    while True:
        # Get audio file path and text query
        print("Insert audio file path:")
        audio_path = input().strip()
        print("Insert query:")
        text = input().strip()

        # Load and process the audio
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != TARGET_SAMPLING_RATE:
            audio = librosa.resample(audio.numpy(), orig_sr=sr, target_sr=TARGET_SAMPLING_RATE)

        # Extract features using the processor
        audio_features = processor(text=text,
                                   audios=audio,
                                   return_tensors="pt",
                                   sampling_rate=TARGET_SAMPLING_RATE,
                                   padding=True,
                                   return_attention_mask=True)

        # Run inference with the model
        with torch.no_grad():
            preds = model(
                is_longer=audio_features['is_longer'].to(DEVICE),
                input_ids=audio_features['input_ids'].squeeze(1).to(DEVICE),
                input_features=audio_features['input_features'].squeeze(1).to(DEVICE),
                attention_mask=audio_features['attention_mask'].to(DEVICE)
            )

        # Calculate cosine similarity
        similarity = cosine_similarity(preds['audio_embeds'], preds['text_embeds'], dim=1)
        print("Cosine similarity:", similarity.item())

        # Ask user if they want to finish or continue
        print("Do you want to continue? (yes/no):")
        continue_choice = input().strip().lower()
        if continue_choice == 'no':
            break


if __name__ == "__main__":
    main()