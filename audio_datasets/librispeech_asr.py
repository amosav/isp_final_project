import librosa
import torch
import torchaudio
from datasets import load_dataset, Dataset
from pyexpat import features
from torch.utils.data import IterableDataset, DataLoader, random_split
from transformers import AutoProcessor
from transformers import ClapModel, ClapProcessor
from transformers.pipelines.base import pad_collate_fn

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




class LibriSpeechCLAPStreamingDataset(IterableDataset):
    def __init__(self, split='train-clean-100', processor=None, max_length=16000):
        """
        Initializes the streaming dataset for finetuning CLAP on LibriSpeech.

        Args:
            split (str): The dataset split to load ('train-clean-100', 'train-clean-360', etc.)
            processor (AutoProcessor): The processor to tokenize/process audio data.
            max_length (int): The maximum length of audio in samples (default is 16,000).
        """
        # self.dataset = load_dataset("librispeech_asr", split=split, streaming=True)
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
            text = f"a {text} making sound"

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
        text = f"a {text} making sound"
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
def get_data_loaders(batch_size=16):
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    streaming_dataset = LibriSpeechCLAPStreamingDataset(split='train.clean.100', processor=processor)
    train_size = int(0.8 * len(streaming_dataset))
    test_size = len(streaming_dataset) - train_size

    # Perform deterministic split
    train_dataset, test_dataset = random_split(streaming_dataset, [train_size, test_size])

    # Create DataLoader objects for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader




