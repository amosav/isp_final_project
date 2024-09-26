import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys

import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoProcessor

from audio_datasets.music_genres_dataset import get_music_genres_data_loaders
from evaluation import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
from audio_datasets.esc50_dataset import get_esc50_data_loaders
from loss import clap_loss
from models.model_utils import get_CLAP_LoRa


class Pipeline:
    def __init__(self,
                 num_epochs,
                 lr,
                 batch_size,
                 r: int = 16,
                 alpha: int = 32,
                 save_path: str = "/content/drive/MyDrive/isp_final_project/",
                 data_type: str = "esc50",
                 add_noise: bool = False):
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = self.init_model(r, alpha)
        self.optimizer, self.scheduler = self.init_optimizer()
        self.train_loader, self.test_loader = self.init_data(data_type, add_noise)
        self.loss = self.init_loss()
        self.save_path = save_path
        self.model_name = f"{data_type}_model_epoch_{self.num_epochs }_lr{self.lr}_a{alpha}r{r}.pt"
        if add_noise:
            self.model_name = f"{data_type}_model_epoch_{self.num_epochs }_lr{self.lr}_a{alpha}r{r}_noise.pt"

    def init_save_path(self):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "checkpoints"), exist_ok=True)

    def init_data(self, data_type="esc50", add_noise=False):
        print(f"Loading {data_type} data..." )
        if data_type == "esc50":
            return get_esc50_data_loaders(True, self.batch_size)
        elif data_type == "music_genres":
            return get_music_genres_data_loaders(True, self.batch_size, add_noise)

    def train(self):
        train_losses = []
        validation_losses = []
        accuracy = []
        # accuracy.append(evaluate(processor, self.model, self.test_loader))
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
            accuracy.append(evaluate(processor, self.model, self.test_loader))

            train_losses.append(train_loss / len(self.train_loader))
            validation_losses.append(validation_loss / len(self.test_loader))
            print(f"Epoch            {epoch + 1}\n"
                  f"Train Loss:      {train_loss / len(self.train_loader)}\n"
                  f"Validation Loss: {validation_loss / len(self.test_loader)}")

        torch.save(self.model.state_dict(), f"{self.save_path}/{self.model_name}")
        self.plot_loss(train_losses, validation_losses, accuracy)

    def plot_loss(self, train_losses, test_losses, accuracy):
        train_losses_x = range(1, len(train_losses) + 1)
        plt.plot(train_losses_x, train_losses, label="Train Loss")
        plt.plot(train_losses_x, test_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.savefig(f"{self.save_path}/{self.model_name}_training_loss.png")
        plt.figure()
        plt.plot(range(1, len(accuracy) + 1), accuracy, label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.title("Accuracy")
        plt.savefig(f"{self.save_path}/{self.model_name}_accuravy.png")

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
    if len(args) == 5:
        lr, r, alpha, data_type, noise = args
        pipeline = Pipeline(num_epochs=30,
                            lr=float(lr),
                            batch_size=32,
                            save_path=save_path,
                            r=int(r),
                            alpha=int(alpha),
                            data_type=data_type,
                            add_noise=noise == "True")
        pipeline.train()
    else:
        print("Usage: python pipeline.py num_epochs lr batch_size")