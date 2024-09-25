import random

import librosa
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import IterableDataset, DataLoader, random_split
from transformers import AutoProcessor
from transformers import ClapProcessor

SEED = 42
torch.manual_seed(SEED)

def collate_fn(batch):
    """
    Custom collate function to handle varying lengths of text and audio.
    This pads all sequences in the batch to the maximum length in the batch for each key.

    Args:
        batch (list): A batch of dictionaries from the dataset.

    Returns:
        dict: A batch where each key is padded to the maximum length.
    """
    # Initialize a dictionary to hold collated data
    collated_batch = {}
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    for key in features[0].keys():
        # Stack all tensors under this key
        values = [b[key].clone().detach() for b in features]

        # Find the max length along axis=1 (sequence length)
        max_length = max(v.size(1) for v in values)

        # Pad each tensor along axis=1 to the maximum length
        padded_values = [torch.nn.functional.pad(v, (0, max_length - v.size(1)), mode='constant', value=0) for v in values]

        # Stack padded tensors along the batch dimension (axis=0)
        collated_batch[key] = torch.stack(padded_values)

    return collated_batch, torch.tensor(labels)

def generate_prompt(data):
    prob = random.uniform(0, 1)
    if prob < 0.25:
        return data
    if prob < 0.5:
        return f"a {data} music"
    if prob < 0.75:
        return f"{data} music playing"
    return f"the sound of {data} music"


class MusicGenresCLAPDataset(IterableDataset):
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
        self.dataset = load_dataset("lewtun/music_genres", split=split)
        self.processor = processor or AutoProcessor.from_pretrained("lewtun/music_genres")
        self.max_length = max_length
        self.target_sampling_rate = 48000
        # genre_values = self.dataset.features['genre'].names
        self.categories_id = {category: i for i, category in enumerate(set(self.dataset['genre']))}


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
            if audio_resampled.size > self.max_length:
                audio_resampled = self.extract_random_segment(audio_resampled)
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
        audio_resampled = self.resample_audio(torch.tensor(audio), original_sampling_rate)
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
def get_music_genres_data_loaders(manipulate_prompt, batch_size=16):
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    if manipulate_prompt:
        train_dataset = MusicGenresCLAPDataset(prompt_function=generate_prompt, split="train", processor=processor)
    else:
        train_dataset = MusicGenresCLAPDataset(prompt_function=lambda x: x, split="train", processor=processor)

    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    # Create DataLoader objects for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader
