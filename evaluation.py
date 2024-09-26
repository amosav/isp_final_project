import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from transformers import ClapModel, AutoProcessor

from audio_datasets.music_genres_dataset import get_music_genres_data_loaders
from visualization import plot_embedding_visualization, plot_cm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def classify_with_cosine_similarity(model, train_loader, caption_embeddings):
    all_predictions = []
    all_true_labels = []
    all_audio_embeddings = []
    for audio_features, true_labels in tqdm(train_loader):
        # Extract audio embeddings from the model
        with torch.no_grad():
            preds = model(
                is_longer=audio_features['is_longer'].to(DEVICE),
                input_ids=audio_features['input_ids'].squeeze(1).to(DEVICE),  # Text input
                input_features=audio_features['input_features'].squeeze(1).to(DEVICE),  # Audio input
                attention_mask=audio_features['attention_mask'].to(DEVICE))

        # Calculate cosine similarity with all caption embeddings
        batch_norm = F.normalize(preds['audio_embeds'], p=2, dim=1)
        reference_set_norm = F.normalize(caption_embeddings, p=2, dim=1)
        similarities = torch.mm(batch_norm, reference_set_norm.t())
        all_audio_embeddings.append(batch_norm)
        best_matches = torch.argmax(similarities, dim=1)

        # Map best matches to predicted labels

        all_predictions.extend(best_matches.tolist())
        all_true_labels.extend(true_labels.tolist())

    all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)

    return all_predictions, all_true_labels, all_audio_embeddings


def evaluate(processor, model, loader, plot_visualization=False):
    model.eval()  # Set the model to evaluation mode
    # Example captions (replace with actual categories from the dataset)
    captions = loader.dataset.dataset.categories_id
    # Generate caption embeddings and store them
    caption_embeddings = []
    for caption, idx in captions.items():
        inputs = processor(text=caption, return_tensors="pt", padding=True, truncation=True)
        for k, v in inputs.items():
          inputs[k] = v.to(DEVICE)
        with torch.no_grad():
            text_embedding = model.get_text_features(**inputs).squeeze(0)  # Get text embeddings

        caption_embeddings.append(text_embedding)
    caption_embeddings = torch.stack(caption_embeddings)
    values_list = list(captions.values())
    pred, true_labels, audio_embeddings = classify_with_cosine_similarity(model, loader, caption_embeddings,)

    accuracy = accuracy_score(true_labels, pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    if plot_visualization:
        cm = confusion_matrix(true_labels, pred)
        plot_cm(captions, cm)
        plot_embedding_visualization(audio_embeddings, true_labels, captions, method="tsne")
    return accuracy


model = ClapModel.from_pretrained("laion/clap-htsat-fused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
model.eval()
train_loader, test_loader = get_music_genres_data_loaders(manipulate_prompt=True, batch_size=16)
evaluate(processor, model, test_loader, plot_visualization=False)