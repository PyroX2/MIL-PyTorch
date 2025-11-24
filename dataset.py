import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2
from typing import Tuple
from image_patcher import ImagePatcher


class MILDataset(Dataset):
    def __init__(self, dataset_path: str, image_patcher: ImagePatcher, transform=None) -> None:
        super().__init__()
        if transform is None:
            transform = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True)
                ])

        self.image_patcher = image_patcher
        self.img_folder_dataset = ImageFolder(dataset_path, transform=transform)

    def __len__(self):
        return len(self.img_folder_dataset)

    def __getitem__(self, index) -> Tuple:
        image, label = self.img_folder_dataset[index]
        c, h, w = image.shape
        self.image_patcher.get_tiles(h, w)
        instances, instances_idx, instances_cords = self.image_patcher.convert_img_to_bag(image)
        return instances, label