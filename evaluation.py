import os

import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from transformers import AutoProcessor, ClapModel

from audio_datasets.music_genres_dataset import get_music_genres_data_loaders, generate_prompt
from models.model_utils import get_CLAP_LoRa
from visualization import plot_embedding_visualization, plot_cm, plot_recall_at_ks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_S = [1, 5, 10]

def classify_with_cosine_similarity(model, train_loader, caption_embeddings):
    all_predictions = []
    all_true_labels = []
    all_audio_embeddings = []
    for audio_features, true_labels in tqdm(train_loader):
        # Extract audio embeddings from the model
        with torch.no_grad():
            audio_embeds = model.get_audio_features(
                is_longer=audio_features['is_longer'].to(DEVICE),
                input_features=audio_features['input_features'].squeeze(1).to(DEVICE),  # Audio input
                attention_mask=audio_features['attention_mask'].to(DEVICE))

        # Calculate cosine similarity with all caption embeddings
        similarities = torch.mm(audio_embeds, caption_embeddings.t())
        all_audio_embeddings.append(audio_embeds)
        best_matches = torch.argmax(similarities, dim=1)

        all_predictions.extend(best_matches.tolist())
        all_true_labels.extend(true_labels.tolist())

    all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)

    return all_predictions, all_true_labels, all_audio_embeddings

def simple_eval_for_training(processor, model, loader):
    model.eval()
    caption_embeddings, captions = get_captions_embeddings(loader, model, processor)
    pred, true_labels, audio_embeddings = classify_with_cosine_similarity(model, train_loader, caption_embeddings)
    accuracy = accuracy_score(true_labels, pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def evaluate(processor, model, loader, model_name):
    model.eval()  # Set the model to evaluation mode
    caption_embeddings, captions = get_captions_embeddings(loader, model, processor)

    pred, true_labels, audio_embeddings = classify_with_cosine_similarity(model, loader, caption_embeddings,)
    cm = confusion_matrix(true_labels, pred)

    recall_at_ks = calculate_recall_at_ks(audio_embeddings, caption_embeddings, true_labels)
    plot_cm(captions, cm, save_path=f"confusion_matrix_{model_name}.png")
    plot_embedding_visualization(audio_embeddings, true_labels, captions, method="tsne", save_path=f"tsne_{model_name}.png")
    return recall_at_ks


def get_captions_embeddings(loader, model, processor):
    captions = loader.dataset.dataset.categories_id
    # Generate caption embeddings and store them
    caption_embeddings = []
    for caption, idx in captions.items():
        prompt = generate_prompt(caption)
        inputs = processor(text=prompt, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embedding = model.get_text_features(
                input_ids=inputs['input_ids'].squeeze(1).to(DEVICE),
                attention_mask=inputs['attention_mask'].to(DEVICE)
            ).squeeze(0)  # Get text embeddings

        caption_embeddings.append(text_embedding)
    caption_embeddings = torch.stack(caption_embeddings)
    return caption_embeddings, captions


def calculate_recall_at_ks(audio_embeddings, caption_embeddings, true_labels):
    audio_embeddings = torch.nn.functional.normalize(audio_embeddings, dim=-1)
    caption_embeddings = torch.nn.functional.normalize(caption_embeddings, dim=-1)
    similarity_matrix = torch.matmul(audio_embeddings, caption_embeddings.T)
    top_k_indices = torch.argsort(similarity_matrix, dim=-1, descending=True)

    # Convert true labels to a tensor for easy comparison
    true_labels = torch.tensor(true_labels, dtype=torch.long, device=DEVICE)

    recalls = {}
    for k in K_S:
        # Check if the true label is in the top k predictions
        top_k_predictions = top_k_indices[:, :k]  # Shape: (num_audio, k)
        correct_predictions = (top_k_predictions == true_labels.unsqueeze(1)).any(dim=1)  # Shape: (num_audio,)
        recall_at_k = correct_predictions.float().mean().item()  # Calculate mean of correct predictions
        recalls[k] = recall_at_k

    return recalls


processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
checkpoint_paths = [
    (r"C:\isp_checkpoints\music_genres_variations\music_genres_model_epoch_12_lr0.0001_a16r16_noise.pt", 16, 16),
    (r"C:\isp_checkpoints\music_genres_variations\music_genres_model_epoch_12_lr0.001_a32r8_noise.pt", 32, 8),
    (r"C:\isp_checkpoints\music_genres_variations\music_genres_model_epoch_12_lr0.001_a32r16_noise.pt", 32, 16),
    (r"C:\isp_checkpoints\music_genres_variations\music_genres_model_epoch_12_lr0.001_a32r32_noise.pt", 32, 32),
    (r"C:\isp_checkpoints\music_genres_variations\music_genres_model_epoch_12_lr0.001_a64r32_noise.pt", 64, 32),
    (r"C:\isp_checkpoints\music_genres_variations\music_genres_model_epoch_12_lr0.0005_a32r16_noise.pt", 32, 16),
]


train_loader, test_loader = get_music_genres_data_loaders(manipulate_prompt=True, batch_size=16)


def evaluate_all_models():
    model_recall = {}
    model = ClapModel.from_pretrained("laion/clap-htsat-fused")
    model.eval()
    model.to(DEVICE)
    model_name = "CLAP"
    model_recall[model_name] = evaluate(processor, model, test_loader, model_name=model_name).values()
    for ckpt, alpha, r in checkpoint_paths:
        model = get_CLAP_LoRa(r=r, alpha=alpha)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model_name = os.path.basename(ckpt).split(".pt")[0]
        model_recall[model_name] = evaluate(processor, model, test_loader, model_name=model_name).values()
    plot_recall_at_ks(model_recall, save_path="recall_at_ks.png")


evaluate_all_models()

