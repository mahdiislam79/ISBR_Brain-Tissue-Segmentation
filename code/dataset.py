import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, list_IDs, image_directory, slice_size=32, target_size=(256, 256), 
                 data_augmentation=True, n_classes=4, mode="train"):
        """
        PyTorch Dataset for loading 2D MRI data.

        Parameters:
        - list_IDs: List of image IDs.
        - image_directory: Directory containing image data.
        - slice_size: Number of slices extracted from a volume (0 means take all slices).
        - target_size: Desired size of each image.
        - data_augmentation: Boolean flag for data augmentation (only used in training).
        - n_classes: Number of segmentation classes.
        - mode: Either "train" or "val" to determine behavior.
        """
        self.list_IDs = list_IDs
        self.image_directory = image_directory
        self.slice_size = slice_size
        self.target_size = target_size
        self.data_augmentation = data_augmentation
        self.n_classes = n_classes
        self.mode = mode  # Determines if it's training or validation

        self.samples = []  # Stores individual slices as samples
        self.labels = []  # Stores corresponding labels as samples
        self.volume_map = []  # Maps volumes to their slice indices (used in validation)
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Prepare the dataset by extracting slices and labels.
        """
        for ID in self.list_IDs:
            # Dummy function for loading slices and labels (replace with actual logic)
            slices, labels = self._load_slices_and_labels(ID)  
            start_idx = len(self.samples)
            self.samples.extend(slices)
            self.labels.extend(labels)
            self.volume_map.append((start_idx, len(self.samples)))

    def _load_slices_and_labels(self, ID):
        """
        Loads slices and labels for a given volume ID using the preprocessing pipeline.

        Parameters:
        - ID: Volume ID to load.

        Returns:
        - slices: List of slices for the volume.
        - labels: Corresponding labels for the slices.
        """
        from preprocessing import Preprocessing  # Ensure the Preprocessing class is imported

        # Initialize the Preprocessing class
        preprocessing = Preprocessing(
            data_aug=self.data_augmentation,
            norm_intensity=True  # Set based on your preference or dataset requirements
        )

        # Process the case using the Preprocessing pipeline
        slices, labels = preprocessing.process_case(
            ID=ID,
            path=self.image_directory,
            number_slices=self.slice_size,
            num_classes=self.n_classes
        )

        # Return slices and labels
        return slices, labels

    def __len__(self):
        if self.slice_size == 0 and self.mode == "val":
            # Validation mode with full volumes
            return len(self.volume_map)
        return len(self.samples)

    def __getitem__(self, index):
        if self.slice_size == 0 and self.mode == "val":
            # Validation mode: Return full volume
            start_idx, end_idx = self.volume_map[index]
            volume_x = torch.tensor(np.stack(self.samples[start_idx:end_idx]), dtype=torch.float32)
            volume_y = torch.tensor(np.stack(self.labels[start_idx:end_idx]), dtype=torch.float32)
            return volume_x, volume_y
        else:
            # Training or validation with slice-based batching
            sample_x = torch.tensor(self.samples[index], dtype=torch.float32)
            label_y = torch.tensor(self.labels[index], dtype=torch.float32)
            return sample_x, label_y

