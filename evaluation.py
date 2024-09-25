import os
import sys
import torch

from models.model_utils import get_CLAP_LoRa

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    sys.path.append('/content/py/isp_final_project') # for colab :)

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import ClapProcessor, ClapModel
from audio_datasets.music_genres_dataset import get_music_genres_data_loaders
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn.functional as F

def evaluate_predictions(predictions, true_labels, label_names):
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    label_dict = {v: k for k, v in label_names.items()}
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    return accuracy



def classify_with_cosine_similarity(model, train_loader, caption_embeddings, caption_labels):
    all_predictions = []
    all_true_labels = []
    all_audio_embeddings = []
    for batch in train_loader:
        audio_features, true_labels = batch  # Unpack the batch (features and true labels)

        # Extract audio embeddings from the model
        with torch.no_grad():
            preds = model(
                is_longer=audio_features['is_longer'].to(DEVICE),
                input_ids=audio_features['input_ids'].squeeze(1).to(DEVICE),  # Text input
                input_features=audio_features['input_features'].squeeze(1).to(DEVICE),  # Audio input
                attention_mask=audio_features['attention_mask'].to(DEVICE))

        # Calculate cosine similarity with all caption embeddings
        batch_norm = F.normalize(preds['audio_embeds'], p=2, dim=1)  # Normalize along the feature dimension (dim=1)
        reference_set_norm = F.normalize(caption_embeddings, p=2, dim=1)
        similarities = torch.mm(batch_norm, reference_set_norm.t())
        all_audio_embeddings.append(batch_norm)
        # Get the index of the most similar caption for each audio sample
        best_matches = torch.argmax(similarities, dim=1)

        # Map best matches to predicted labels

        all_predictions.extend(best_matches.tolist())
        all_true_labels.extend(true_labels.tolist())

    all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0)

    return all_predictions, all_true_labels, all_audio_embeddings


def evaluate(processor, model, loader):
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
    pred, true_labels, audio_embeddings = classify_with_cosine_similarity(model, loader, caption_embeddings, values_list)
    return evaluate_predictions(pred, true_labels, captions)
    plot_embedding_visualization(audio_embeddings, true_labels,captions, method="tsne")


def plot_embedding_visualization(embeddings, labels, label_names, method="pca", n_components=2):
    """
    Visualizes high-dimensional embeddings using PCA or t-SNE and adds label names directly on the plot.

    Parameters:
        embeddings: Numpy array of the embeddings.
        labels: True labels for each embedding.
        label_names: Dictionary of label names {label_index: label_name}.
        method: Either 'pca' or 'tsne' for dimensionality reduction.
        n_components: Number of dimensions for reduction (2 for visualization).
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = "PCA Visualization of Audio Embeddings"
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=30, n_iter=300)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = "t-SNE Visualization of Audio Embeddings"
    label_names = {v: k for k, v in label_names.items()}
    # Create a color palette to match each label to a color
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hsv", len(unique_labels))
    color_mapping = {label: palette[i] for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(12, 8))

    # Scatter plot for each label
    for label in unique_labels:
        idx = np.where(np.array(labels) == label)
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], color=color_mapping[label], s=50)

        # Calculate the center of the cluster and place the label name there
        x_mean = np.mean(reduced_embeddings[idx, 0])
        y_mean = np.mean(reduced_embeddings[idx, 1])
        plt.text(x_mean, y_mean, label_names[label], fontsize=12, weight='bold',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


# Initialize the processor and model
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
model = ClapModel.from_pretrained("laion/clap-htsat-fused")
# model = get_CLAP_LoRa(16, 32)

# model.load_state_dict(torch.load("esc50_model_epoch_6_lr0.0001_a32r16.pt", map_location=torch.device('cpu')))
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(DEVICE)
model.eval()
#
# train_loader, test_loader =get_esc50_data_loaders(True)
train_loader, test_loader =get_music_genres_data_loaders(False)
evaluate(processor, model, test_loader)

# for i in train_loader:
#     # print(i)
#     a = model(
#         is_longer=i['is_longer'],
#         input_ids=i['input_ids'].squeeze(1),  # Text input
#         input_features=i['input_features'].squeeze(1),  # Audio input
#         attention_mask=i['attention_mask']
#     )
#     break