import librosa
import numpy as np
from sklearn.decomposition import PCA
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
