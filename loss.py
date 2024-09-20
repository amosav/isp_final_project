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
    """        # Labels are simply the indices of the batch
    batch_size = audio_embeddings.shape[0]
    labels = torch.arange(batch_size, device=audio_embeddings.device)

    # Compute cross-entropy loss for image-to-text and text-to-image
    loss_a2t = F.cross_entropy(audio_embeddings, labels)
    loss_t2a = F.cross_entropy(text_embeddings, labels)

    # The total loss is the average of the two
    total_loss = (loss_a2t + loss_t2a) / 2
    return total_loss
