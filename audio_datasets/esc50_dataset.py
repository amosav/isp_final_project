import random

import librosa
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader, random_split
from transformers import AutoProcessor
from transformers import ClapProcessor

from audio_datasets.data_augmentations import collate_fn

SEED = 42
torch.manual_seed(SEED)

def generate_prompt(data):
    prob = random.uniform(0, 1)
    if prob < 0.25:
        return data
    if prob < 0.5:
        return f"a {data} making sound"
    if prob < 0.75:
        return f"this is a {data}"
    return f"the sound of a {data}"



class esc50CLAPStreamingDataset(IterableDataset):
    def __init__(self, prompt_function, split='train', processor=None, max_length=16000):
        """
        Initializes the streaming dataset for finetuning CLAP on LibriSpeech.

        Args:
            split (str): The dataset split to load ('train-clean-100', 'train-clean-360', etc.)
            processor (AutoProcessor): The processor to tokenize/process audio data.
            max_length (int): The maximum length of audio in samples (default is 16,000).
        """
        # self.dataset = load_dataset("librispeech_asr", split=split, streaming=True)
        self.prompt_function = prompt_function
        self.dataset = load_dataset("ashraq/esc50", split="train", streaming=False)
        self.processor = processor or AutoProcessor.from_pretrained("laion/clap-htsat-fused")
        self.max_length = max_length
        self.categories_id = {category: i for i, category in enumerate(set(self.dataset['category']))}
        self.target_sampling_rate = 48000

    def resample_audio(self, audio, original_sampling_rate):
        if original_sampling_rate != self.target_sampling_rate:
            audio = librosa.resample(audio.numpy(), orig_sr=original_sampling_rate, target_sr=self.target_sampling_rate)
        return audio

    def __iter__(self):
        """
        Yields each item in the dataset.

        Returns:
            dict: A dictionary with processed audio and corresponding text.
        """
        for sample in self.dataset:
            # Process the audio and text
            audio = sample['audio']['array']
            original_sampling_rate = sample['audio']['sampling_rate']
            audio_resampled = self.resample_audio(torch.tensor(audio), original_sampling_rate)

            text = sample['category']
            text = self.prompt_function(text)

            # Process the audio using the CLAP processor
            audio_features = self.processor(
                audios=audio_resampled,
                text=text,
                return_tensors="pt",
                sampling_rate=sample['audio']['sampling_rate'],
                truncation=True,
                padding=True,
                return_attention_mask=True
            )

            # Yield the processed sample
            yield {
                **audio_features,
            }
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        audio = data['audio']['array']
        original_sampling_rate = data['audio']['sampling_rate']
        audio_resampled = self.resample_audio(torch.tensor(audio), original_sampling_rate)
        text = data['category']
        text = self.prompt_function(text)
        audio_features = self.processor(
            audios=audio_resampled,
            text=text,
            return_tensors="pt",
            sampling_rate=self.target_sampling_rate,
            truncation=True,
            padding=True,
            return_attention_mask=True
        )
        return audio_features, self.categories_id[data['category']]


# Example usage
def get_esc50_data_loaders(manipulate_prompt, batch_size=16):
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    if manipulate_prompt:
        streaming_dataset = esc50CLAPStreamingDataset(prompt_function=generate_prompt, processor=processor)
    else:
        streaming_dataset = esc50CLAPStreamingDataset(prompt_function=lambda x: x, processor=processor)

    train_size = int(0.8 * len(streaming_dataset))
    test_size = len(streaming_dataset) - train_size

    # Perform deterministic split
    train_dataset, test_dataset = random_split(streaming_dataset, [train_size, test_size])

    # Create DataLoader objects for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    return train_loader, test_loader




