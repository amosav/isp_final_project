import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('your_audio_file.wav', sr=None)

# Extract MFCC features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Flatten the features for PCA
mfcc_flattened = mfcc.T  # Transpose to make each row a feature vector

# Assume you have a matrix 'embeddings' where each row is an embedding vector
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization or as required
embeddings_pca = pca.fit_transform(mfcc_flattened)

print("Explained variance by each component:", pca.explained_variance_ratio_)

plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
plt.title('PCA of Audio Embeddings')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


# T-sne
y, sr = librosa.load('your_audio_file.wav', sr=None)

# Extract MFCC features (13 coefficients)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Transpose to get feature vectors as rows
embedding_vector = mfcc.T
# Example embedding matrix: each row corresponds to an audio file embedding
# Assume 'embeddings' is a numpy array where each row is an embedding vector
# Replace 'embedding_vector' with the matrix for all audio files.
embeddings = np.array([embedding_vector, ...])  # Add all embeddings

# Apply t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the 2D t-SNE result
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title('t-SNE on Audio Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

