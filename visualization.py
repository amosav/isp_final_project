import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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


def plot_cm(captions, cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=captions, yticklabels=captions)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
