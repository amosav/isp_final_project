import torch
import torch.nn.functional as F

def contrastive_loss(embedding1, embedding2, label, margin=1.0):
    # Calculate Euclidean distance
    distance = F.pairwise_distance(embedding1, embedding2)
    # Contrastive Loss: L = (1-label) * 0.5 * distance^2 + label * 0.5 * max(0, margin - distance)^2
    loss = (1 - label) * 0.5 * torch.pow(distance, 2) + label * 0.5 * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()
