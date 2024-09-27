import numpy as np
import torch
import random
import torchaudio
from scipy import signal
import librosa
import torchaudio.transforms as T
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


def random_crop(audio, max_length, crop_length):
    if crop_length >= max_length:
        return audio
    start = random.randint(0, max_length - crop_length)
    return audio[start:start + crop_length]


def add_colored_noise(audio, snr_db, color='pink'):
    white_noise = np.random.randn(len(audio))
    # Apply filter for pink noise
    if color == 'pink':
        b, a = signal.butter(1, 0.2, btype='low')  # Low-pass filter for pink noise
        noise = signal.lfilter(b, a, white_noise)
    else:
        noise = white_noise  # Default white noise

    # Adjust noise to desired SNR
    audio_power = torch.mean(audio ** 2)
    noise_power = audio_power / (10 ** (snr_db / 10))
    noise = torch.tensor(noise)
    noise = noise * torch.sqrt(noise_power / torch.mean(noise ** 2))

    # Add noise to audio
    noisy_audio = audio + noise
    return np.clip(noisy_audio, -1.0, 1.0)

from torchaudio.transforms import TimeMasking, FrequencyMasking

def spec_augment(audio, sr, time_mask_param=30, freq_mask_param=15):
    # Convert audio to spectrogram using librosa
    mel_spectrogram_transform = T.MelSpectrogram( sample_rate=sr,n_fft=1024, win_length=1024, hop_length=512, n_mels=80)
    time_masking = TimeMasking(time_mask_param=time_mask_param)
    freq_masking = FrequencyMasking(freq_mask_param=freq_mask_param)
    spec_augmented = freq_masking(time_masking(mel_spectrogram_transform(audio.float())))
    mel_to_spec_transform = T.InverseMelScale(n_stft=(1024 // 2 + 1), n_mels=80)
    griffin_lim_transform = T.GriffinLim(n_fft=1024, hop_length=512)
    spectrogram = mel_to_spec_transform(spec_augmented)
    return  griffin_lim_transform(spectrogram)



