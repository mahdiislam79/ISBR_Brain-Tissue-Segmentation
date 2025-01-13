import os
from dataloader import get_dataloaders

class Configuration:
    def __init__(self, path_data, target_size=(256, 256), slice_size=32, batch_size=16,  n_classes=4):
        """
        Configuration class to manage dataset and dataloader setups.

        Parameters:
        - path_data: Path to the dataset directory.
        - target_size: Desired size of each image (height, width).
        - batch_size: Number of samples in a batch.
        - mini_batch_size: Number of slices extracted from a single volume.
        - n_classes: Number of classes in segmentation.
        """
        self.path_data = path_data
        self.target_size = target_size
        self.batch_size = batch_size
        self.slice_size = slice_size
        self.n_classes = n_classes

        # Predefined splits for training and validation
        self.train_ids = [
            'IBSR_01', 'IBSR_03', 'IBSR_04', 'IBSR_05', 'IBSR_06', 
            'IBSR_07', 'IBSR_08', 'IBSR_09', 'IBSR_16', 'IBSR_18'
        ]
        self.val_ids = ['IBSR_11', 'IBSR_12', 'IBSR_13', 'IBSR_14', 'IBSR_17']

    def create_dataloaders(self):
        """
        Creates training and validation DataLoaders.

        Returns:
        - train_loader: DataLoader for the training set.
        - val_loader: DataLoader for the validation set.
        """
        train_loader, val_loader = get_dataloaders(
            self.path_data, self.train_ids, self.val_ids,
            slice_size=self.slice_size,
            batch_size=self.batch_size,
            target_size=self.target_size,
            n_classes=self.n_classes
        )
        return train_loader, val_loader

