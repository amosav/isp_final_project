import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from audio_datasets.librispeech_asr import get_wrapped_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from loss import contrastive_loss
from models.naive_model import AudioEmbeddingModel


class Pipeline:
    def __init__(self,
                 num_epochs,
                 lr,
                 batch_size,):
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = self.init_model()
        self.optimizer, self.scheduler = self.init_optimizer()
        self.train_loader = self.init_data()
        self.loss = self.init_loss()

    def init_data(self):
        return get_wrapped_dataset(batch_size=self.batch_size)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0 # Initialize loss
            for batch in tqdm(self.train_loader):
                input_values = batch["input_values"].to(DEVICE)
                speaker_ids = batch["speaker_ids"]

                # Generate positive and negative pairs
                batch_size = input_values.size(0)
                half_batch = batch_size // 2
                anchor, positive = input_values[:half_batch], input_values[half_batch:]

                # Positive pairs: same speaker
                labels = torch.tensor(
                    [1 if speaker_ids[i] == speaker_ids[half_batch + i] else 0 for i in range(half_batch)],
                    dtype=torch.float32).to(DEVICE)

                # Compute embeddings
                embedding_anchor = self.model(anchor)
                embedding_positive = self.model(positive)

                # Compute contrastive loss
                loss = self.loss(embedding_anchor, embedding_positive, labels)
                train_loss += loss.item()
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            print(f"Epoch {epoch + 1}, Loss: {train_loss / len(self.train_loader)}")

    def init_model(self):
        return AudioEmbeddingModel()

    def init_loss(self):
        return contrastive_loss

    def init_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return optimizer, scheduler

pipeline = Pipeline(num_epochs=10, lr=1e-3, batch_size=32)
pipeline.train()
