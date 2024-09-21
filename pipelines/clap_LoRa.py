import os
import sys

import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    sys.path.append('/content/py/isp_final_project') # for colab :)

from audio_datasets.librispeech_asr import get_data_loaders
from loss import clap_loss
from models.model_utils import get_CLAP_LoRa


class Pipeline:
    def __init__(self,
                 num_epochs,
                 lr,
                 batch_size,
                 r: int = 16,
                 alpha: int = 32,
                 save_path: str = "/content/drive/MyDrive/isp_final_project/"):
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = self.init_model(r, alpha)
        self.optimizer, self.scheduler = self.init_optimizer()
        self.train_loader, self.test_loader = self.init_data()
        self.loss = self.init_loss()
        self.save_path = save_path
        self.model_name = f"model_epoch_{self.num_epochs }_lr{self.lr}_a{alpha}r{r},.pt"

    def init_save_path(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "checkpoints"), exist_ok=True)

    def init_data(self):
        return get_data_loaders(True, self.batch_size)

    def train(self):
        train_losses = []
        validation_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0 # Initialize loss
            validation_loss = 0
            for batch, categories in tqdm(self.train_loader):
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
            for batch, categories in tqdm(self.test_loader):
                preds = self.model(
                    is_longer=batch['is_longer'].to(DEVICE),
                    input_ids=batch['input_ids'].squeeze(1).to(DEVICE),  # Text input
                    input_features=batch['input_features'].squeeze(1).to(DEVICE),  # Audio input
                    attention_mask=batch['attention_mask'].to(DEVICE)
                )
                loss = self.loss(preds['audio_embeds'], preds['text_embeds'])
                validation_loss += loss.item()

            train_losses.append(train_loss / len(self.train_loader))
            validation_losses.append(validation_loss / len(self.test_loader))
            print(f"Epoch            {epoch + 1}\n"
                  f"Train Loss:      {train_loss / len(self.train_loader)}\n"
                  f"Validation Loss: {validation_loss / len(self.test_loader)}")

        torch.save(self.model.state_dict(), f"{self.save_path}/{self.model_name}")
        self.plot_loss(train_losses, validation_losses)

    def plot_loss(self, train_losses, test_losses):
        train_losses_x = range(1, len(train_losses) + 1)
        plt.plot(train_losses_x, train_losses, label="Train Loss")
        plt.plot(train_losses_x, test_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.savefig(f"{self.save_path}/{self.model_name}_training_loss.png")

    def init_model(self, r, alpha):
        return get_CLAP_LoRa(r, alpha)

    def init_loss(self):
        return clap_loss

    def init_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        return optimizer, scheduler


if __name__ == '__main__':
    save_path = "/content/drive/MyDrive/isp_final_project/"
    args = sys.argv[1:]
    if len(args) == 3:
        lr, r, alpha = args
        pipeline = Pipeline(num_epochs=12,
                            lr=float(lr),
                            batch_size=32,
                            save_path=save_path,
                            r=int(r),
                            alpha=int(alpha))
        pipeline.train()
    else:
        print("Usage: python pipeline.py num_epochs lr batch_size")