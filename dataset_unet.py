import os
import numpy as np
from torch.utils.data import Dataset
import torch


class organ_segmentation_dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.load(img_path)
        mask = np.load(mask_path)

        # Normaliza la imagen a [0, 1] si no ya está normalizada
        image_max = np.amax(image)
        if image_max > 1.0:
            image = image / image_max

        mask = mask.astype(np.int64)  # Las máscaras deben ser de tipo long para el cálculo de pérdida

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# Example of a simple transform function
def transform1(image, mask):
    # Convert numpy arrays to PyTorch tensors
    image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
    mask = torch.from_numpy(mask)
    return image, mask