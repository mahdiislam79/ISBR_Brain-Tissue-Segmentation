import os
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessing import Preprocessing

class MRIDataset(Dataset):
    def __init__(self, list_IDs, image_directory, minibatch_size=32, target_size=(256, 256), 
                 data_augmentation=True, n_classes=1):
        """
        PyTorch Dataset for loading 2D MRI data.

        Parameters:
        - list_IDs: List of image IDs.
        - image_directory: Directory containing image data.
        - minibatch_size: Number of slices in a mini-batch.
        - target_size: Desired size of each image.
        - data_augmentation: Boolean flag for data augmentation.
        - n_classes: Number of classes in segmentation.
        """
        self.image_directory = image_directory
        self.list_IDs = list_IDs
        self.minibatch_size = minibatch_size
        self.target_size = target_size
        self.n_classes = n_classes
        self.data_augmentation = data_augmentation
        self.preprocessing = Preprocessing(data_aug=data_augmentation, norm_intensity=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        number_slices = self.minibatch_size
        x, y = self.preprocessing.process_case(ID, self.image_directory, number_slices, self.n_classes)

        # Add channel dimension for compatibility
        x = np.expand_dims(x, axis=1)  # (Slices, 1, H, W)
        y = np.expand_dims(y, axis=1)  # (Slices, 1, H, W)

        # Convert to PyTorch tensors
        batch_x = torch.tensor(x, dtype=torch.float32)
        batch_y = torch.tensor(y, dtype=torch.float32)

        return batch_x, batch_y
