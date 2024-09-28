import torch
from datasets import load_dataset
from transformers import ClapProcessor

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Load the ESC-50 dataset and the CLAP processor
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

# Dataset loading
music_dataset = load_dataset("lewtun/music_genres_small", split="train")
print(1)