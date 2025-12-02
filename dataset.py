import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2
from typing import Tuple
from image_patcher import ImagePatcher
import os
import numpy as np
from PIL import Image


class MILDataset(Dataset):
    def __init__(self, dataset_path: str, image_patcher: ImagePatcher, dirs_with_classes: dict = None, transform=None) -> None:
        super().__init__()

        # Prepare image transforms
        if transform is None:
            self.transform = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True)
                ])
        else:
            self.transform = transform

        # Init image patcher
        self.image_patcher = image_patcher

        # If user didn't specify subset of directories use all
        if dirs_with_classes is None:
            dirs_with_classes = {}
            for class_idx, dir_name in enumerate(os.listdir(dataset_path)):
                dirs_with_classes[dir_name] = class_idx

        # For each directory get all image paths and assign label to them
        self.image_paths = []
        self.labels = []
        for dir_name, label in dirs_with_classes.items():
            # Path to class directory
            class_path = os.path.join(dataset_path, dir_name)

            class_image_paths = [os.path.join(class_path, img_filename) for img_filename in os.listdir(class_path)]
            class_labels = [label for _ in range(len(class_image_paths))]

            self.image_paths.extend(class_image_paths)
            self.labels.extend(class_labels)

        # Shuffle the data
        p_list = np.random.permutation(len(self.image_paths))
        self.image_paths = [self.image_paths[p] for p in p_list]
        self.labels = [self.labels[p] for p in p_list]

        self.labels = torch.tensor(self.labels, dtype=torch.long)     # Convert to tensor

        self.classes = list(set(dirs_with_classes.values()))
        self.dirs_with_classes = dirs_with_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> Tuple:
        image_path, label = self.image_paths[index], self.labels[index]

        image = Image.open(image_path)

        # Normalization
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)      # Add channel dimension to grayscale image
        image = image.repeat(repeats=3, axis=-1)    # Grayscale to RGB
        image = (image - image.min()) / (image.max() - image.min())

        image = self.transform(image)

        c, h, w = image.shape
        self.image_patcher.get_tiles(h, w)
        instances, instances_idx, instances_cords = self.image_patcher.convert_img_to_bag(image)
        return instances, label