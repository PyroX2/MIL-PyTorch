import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler, Subset
import torch.distributed as dist


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


def create_dataloader(dataset, batch_size, num_workers, is_ddp, rank=0, world_size=1, shuffle=True):
    """
    Create a DataLoader for the given dataset, handling both distributed and non-distributed settings.

    Batch size is the batch size used per rank in distributed mode.
    """

    targets = torch.tensor(dataset.img_folder_dataset.targets, dtype=torch.long) # Get all targets from the dataset

    if not is_ddp:
        if shuffle:
            # Use weighted random sampler to handle class imbalance
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
    else:
        if rank == 0:
            all_idx = torch.arange(len(dataset)).numpy()
        else:
            all_idx = None
        
        # Create object list to hold all indices and broadcast to all ranks
        # broadcast_object_list is used for Python objects (lists, numpy arrays, etc.)
        obj_list = [all_idx]

        # Broadcast all indices to all ranks
        dist.broadcast_object_list(obj_list, src=0)
        all_idx = obj_list[0]
        
        subset = Subset(dataset, all_idx)

        sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank, shuffle=shuffle)

        dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, sampler=sampler)

        return dataloader
