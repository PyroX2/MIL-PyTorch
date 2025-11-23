import torch


def collate_fn(batch):
    # Aquires important dimensions
    batch_size = len(batch)
    c, h, w = batch[0][0].shape[1:]
    max_bag_length = max([len(x) for x, y in batch])
    
    # Initializing placeholders for features and labels
    features = torch.zeros((batch_size*max_bag_length, c, h, w))
    labels = torch.zeros((len(batch)), dtype=torch.long)

    # Masking placeholder, mask = 1 for valid instances, 0 for padded instances
    masks = torch.zeros((batch_size*max_bag_length))

    # Empty image used for padding
    pad_image = torch.zeros((1, c, h, w))

    for i, (x, y) in enumerate(batch):
        n_instances, c, h, w = x.shape

        # Set features and labels
        features[i*max_bag_length:(i*max_bag_length+n_instances)] = x
        features[(i*max_bag_length+n_instances):(i+1)*max_bag_length] = pad_image.expand((max_bag_length-n_instances, c, h, w))

        masks[i*max_bag_length:(i*max_bag_length+n_instances)] = 1
        labels[i] = y

    return features, labels, masks
