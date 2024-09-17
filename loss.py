import torch
import torch.nn.functional as F

def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    distance = F.pairwise_distance(embedding1, embedding2)
    # Contrastive Loss: L = (1-label) * 0.5 * distance^2 + label * 0.5 * max(0, margin - distance)^2
    loss = (1 - label) * 0.5 * torch.pow(distance, 2) + label * 0.5 * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

def clap_loss(audio_embeddings, text_embeddings):
    """
    Computes the CLAP loss as the cosine similarity between audio and text embeddings.

    Args:
        audio_embeddings: Tensor containing the audio embeddings.
        text_embeddings: Tensor containing the text embeddings.

    Returns:
        A scalar tensor representing the CLAP loss.
    """
    # Normalize embeddings to have unit length (for cosine similarity)
    audio_embeddings = F.normalize(audio_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    # Compute the cosine similarity
    cosine_sim = F.cosine_similarity(audio_embeddings, text_embeddings, dim=1)

    # The loss is 1 - mean cosine similarity, encouraging the model to maximize similarity
    loss = 1 - cosine_sim.mean()

    return loss
