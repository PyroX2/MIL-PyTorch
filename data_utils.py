import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler


# Given a batch of bags with varying number of instances, collate them into a single tensor with padding. Returns features, labels, masks, and bags length.
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

    return features, labels, masks, max_bag_length


def create_dataloader(dataset, batch_size, num_workers, is_ddp, shuffle=True):
    if not is_ddp:
        if shuffle:
            # Use weighted random sampler to handle class imbalance
            targets = torch.tensor(dataset.img_folder_dataset.targets, dtype=torch.long) # Get all targets from the dataset
            counts = torch.unique(targets, return_counts=True) # Count occurrences of each class
            
            # Calculate weights for each class
            weights = 1.0 / counts[1].float()
            samples_weights = weights[targets]

            # Create sampler
            weighted_sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
            
            # Create dataloader
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, sampler=weighted_sampler)

            return dataloader
        else:
            return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=False)