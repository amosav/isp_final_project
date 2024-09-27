import os
import random

import librosa
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader, random_split
from transformers import AutoProcessor
from transformers import ClapProcessor

from audio_datasets.data_augmentations import collate_fn, random_crop, add_colored_noise, spec_augment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 42
torch.manual_seed(SEED)
MIN_SECOND_LENGTH = 2
MAX_SECOND_LENGTH = 10

def generate_prompt(data):
    possible_prompts = [f"the sound of {data} music",
                        f"this is {data} music",
                        f"{data} music",
                        f"{data} music playing",
                        f"{data}"]
    return random.choice(possible_prompts)



class MusicGenresCLAPDataset(IterableDataset):
    def __init__(self, prompt_function, split='train', processor=None, augment=False):
        """
        Initializes the streaming dataset for finetuning CLAP on LibriSpeech.

        Args:
            split (str): The dataset split to load ('train-clean-100', 'train-clean-360', etc.)
            processor (AutoProcessor): The processor to tokenize/process audio data.
        """
        # self.dataset = load_dataset("librispeech_asr", split=split, streaming=True)
        self.prompt_function = prompt_function
        self.dataset = load_dataset("lewtun/music_genres_small", split=split)
        self.processor = processor or AutoProcessor.from_pretrained("lewtun/music_genres_small")
        self.target_sampling_rate = 48000
        self.snr_range = (10, 30)
        self.color_noise = ['white', 'pink']
        self.color_noise_prob = 0.15
        self.augment = augment
        self.color_augment_prob = 0.4
        # genre_values = self.dataset.features['genre'].names
        self.categories_id = {category: i for i, category in enumerate(set(self.dataset['genre']))}

    def augment_audio(self, audio, original_sampling_rate):
        if original_sampling_rate != self.target_sampling_rate:
            audio = librosa.resample(audio.numpy(), orig_sr=original_sampling_rate, target_sr=self.target_sampling_rate)
        length = random.uniform(MIN_SECOND_LENGTH, MAX_SECOND_LENGTH) * self.target_sampling_rate

        audio = random_crop(audio, len(audio), int(length))

        if random.uniform(0, 1) <self.color_noise_prob:
            snr_db = random.uniform(*self.snr_range)
            color = random.choice(self.color_noise)
            audio = add_colored_noise(torch.tensor(audio), snr_db, color)
        if random.uniform(0, 1) <self.color_augment_prob:
            audio = spec_augment(audio, self.target_sampling_rate, time_mask_param=30, freq_mask_param=15)
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
            audio_resampled = self.augment_audio(torch.tensor(audio), original_sampling_rate)
            text = sample['genre']
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
        audio_resampled = self.augment_audio(torch.tensor(audio), original_sampling_rate)
        text = data['genre']
        text = self.prompt_function(text)
        audio_features = self.processor(
            audios=audio_resampled,
            text=text,
            return_tensors="pt",
            sampling_rate=self.target_sampling_rate,
            padding=True,
            return_attention_mask=True
        )
        return audio_features, self.categories_id[data['genre']]


    def extract_random_segment(self, audio):
        """Randomly extract a 5-second segment from the audio."""
        max_start = max(0, len(audio) - self.max_length)
        start = random.randint(0, max_start)
        return audio[start:start + self.max_length]


# Example usage
def get_music_genres_data_loaders(manipulate_prompt, batch_size=16, add_noise=False):
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    if manipulate_prompt:
        train_dataset = MusicGenresCLAPDataset(prompt_function=generate_prompt,
                                               split="train",
                                               processor=processor,
                                               augment=add_noise)
    else:
        train_dataset = MusicGenresCLAPDataset(prompt_function=lambda x: x,
                                               split="train",
                                               processor=processor,
                                               augment=add_noise)
    print("Noise is ", add_noise)
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    # Create DataLoader objects for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    return train_loader, test_loader
