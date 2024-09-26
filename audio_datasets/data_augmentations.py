import torch


def collate_fn(batch):
    """
    Custom collate function to handle varying lengths of text and audio.
    This pads all sequences in the batch to the maximum length in the batch for each key.

    Args:
        batch (list): A batch of dictionaries from the dataset.

    Returns:
        dict: A batch where each key is padded to the maximum length.
    """
    # Initialize a dictionary to hold collated data
    collated_batch = {}
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    for key in features[0].keys():
        # Stack all tensors under this key
        values = [b[key].clone().detach() for b in features]

        # Find the max length along axis=1 (sequence length)
        max_length = max(v.size(1) for v in values)

        # Pad each tensor along axis=1 to the maximum length
        padded_values = [torch.nn.functional.pad(v, (0, max_length - v.size(1)), mode='constant', value=0) for v in values]

        # Stack padded tensors along the batch dimension (axis=0)
        collated_batch[key] = torch.stack(padded_values)

    return collated_batch, torch.tensor(labels)
