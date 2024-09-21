import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from audio_datasets.librispeech_asr import get_data_loaders
from loss import clap_loss
from models.model_utils import get_CLAP_LoRa

import sys
sys.path.append('/content/py/isp_final_project') # for colab :)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.train_loader, self.test_loader = self.init_data()
        self.loss = self.init_loss()

    def init_data(self):
        return get_data_loaders(self.batch_size)

    def train(self):
        train_losses = []
        validation_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0 # Initialize loss
            for batch in tqdm(self.train_loader):
                preds = self.model(
                   is_longer=batch['is_longer'].to(DEVICE),
                   input_ids=batch['input_ids'].squeeze(1).to(DEVICE),  # Text input
                   input_features=batch['input_features'].squeeze(1).to(DEVICE),  # Audio input
                   attention_mask=batch['attention_mask'].to(DEVICE)
               )
                loss = self.loss(preds['audio_embeds'], preds['text_embeds'])
                train_loss += loss.item()
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
            self.model.eval()
            for batch in tqdm(self.test_loader):
                preds = self.model(
                   is_longer=batch['is_longer'].to(DEVICE),
                   input_ids=batch['input_ids'].squeeze(1).to(DEVICE),  # Text input
                   input_features=batch['input_features'].squeeze(1).to(DEVICE),  # Audio input
                   attention_mask=batch['attention_mask'].to(DEVICE)
               )
                loss = self.loss(preds['audio_embeds'], preds['text_embeds'])
                validation_losses.append(loss.item())
            torch.save(self.model.state_dict(), f"model_epoch_{epoch + 1}.pt")

            train_losses.append(train_loss / len(self.train_loader))
            print(f"Epoch {epoch + 1}, Loss: {train_loss / len(self.train_loader)}")
            print(f"Validation Loss: {sum(validation_losses) / len(validation_losses)}")

        self.plot_loss(train_losses, validation_losses)

    def plot_loss(self, train_losses, test_losses):
        train_losses_x = range(1, len(train_losses) + 1)
        plt.plot(train_losses_x, train_losses, label="Train Loss")
        plt.plot(train_losses_x, test_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.savefig("training_loss.png")

    def init_model(self):
        return get_CLAP_LoRa()

    def init_loss(self):
        return clap_loss

    def init_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return optimizer, scheduler


if __name__ == '__main__':
    pipeline = Pipeline(num_epochs=10, lr=1e-4, batch_size=16)
    pipeline.train()