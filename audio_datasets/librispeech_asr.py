import librosa
import torch
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader



# Preprocessing function
def preprocess(batch):
    # Load audio and resample to 16 kHz
    audio = batch["audio"]
    y = audio["array"]
    sr = audio["sampling_rate"]

    # Extract 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)

    # Update batch with MFCCs and other information
    batch["input_values"] = mfccs
    batch["input_length"] = mfccs.shape[1]  # Length is the number of frames
    batch["speaker_id"] = batch["speaker_id"]
    batch["text"] = batch["text"]
    return batch


# Define a custom collate function to handle varying audio lengths
def collate_fn(batch):
    input_values = [item["input_values"].T for item in batch]  # Transpose to (time, n_mfcc)
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0)

    # Transpose back to (n_mfcc, time) after padding
    input_values = input_values.transpose(1, 2)

    input_lengths = torch.tensor([item["input_length"] for item in batch])

    # Fetch transcriptions and speaker IDs
    labels = [item["text"] for item in batch]  # Transcriptions
    speaker_ids = [item["speaker_id"] for item in batch]  # Speaker IDs

    return {"input_values": input_values, "input_lengths": input_lengths, "labels": labels, "speaker_ids": speaker_ids}

def get_wrapped_dataset(batch_size):
    # Load a small portion of the dataset
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100[:250]")
    # Apply the preprocessing
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(preprocess)
    # Convert to PyTorch format
    dataset.set_format(type="torch", columns=["input_values", "speaker_id", "text", "input_length"])
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return dataloader


# Iterate through the DataLoader
# for batch in dataloader:
#     input_values = batch["input_values"]
#     input_lengths = batch["input_lengths"]
#     labels = batch["labels"]
#     speaker_ids = batch["speaker_ids"]
#     print(input_values.shape, input_lengths, labels, speaker_ids)
