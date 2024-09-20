
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoProcessor
from transformers import ClapModel, ClapProcessor

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

    for key in batch[0].keys():
        # Stack all tensors under this key
        values = [torch.tensor(b[key]) for b in batch]

        # Find the max length along axis=1 (sequence length)
        max_length = max(v.size(1) for v in values)

        # Pad each tensor along axis=1 to the maximum length
        padded_values = [torch.nn.functional.pad(v, (0, max_length - v.size(1)), mode='constant', value=0) for v in values]

        # Stack padded tensors along the batch dimension (axis=0)
        collated_batch[key] = torch.stack(padded_values)

    return collated_batch




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
        self.dataset = load_dataset("ashraq/esc50", split="train", streaming=True)
        self.processor = processor or AutoProcessor.from_pretrained("laion/clap-htsat-fused")
        self.max_length = max_length

    def resample_audio(self, audio, original_sampling_rate):
        if original_sampling_rate != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=self.target_sampling_rate)
            return resampler(audio)
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


# Example usage
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
streaming_dataset = LibriSpeechCLAPStreamingDataset(split='train.clean.100', processor=processor)
data_loader = DataLoader(streaming_dataset, batch_size=16, collate_fn=collate_fn)
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
for i in data_loader:
    # print(i)
    a = model(
        is_longer=i['is_longer'],
        input_ids=i['input_ids'].squeeze(1),  # Text input
        input_features=i['input_features'].squeeze(1),  # Audio input
        attention_mask=i['attention_mask']
    )
    break