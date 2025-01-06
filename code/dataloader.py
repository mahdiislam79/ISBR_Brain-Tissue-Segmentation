import os
from dataset import MRIDataset
from torch.utils.data import DataLoader

def get_dataloaders(path_data, train_ids, val_ids, slice_size=32, batch_size=16, target_size=(256, 256), n_classes=4):
    """
    Creates DataLoaders for training and validation datasets.

    Parameters:
    - path_data: Path to the data directory.
    - train_ids: List of training image IDs.
    - val_ids: List of validation image IDs.
    - slice_size: Number of slices in a mini-batch.
    - target_size: Desired size of each image.
    - n_classes: Number of segmentation classes.

    Returns:
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    """
    # Create training dataset
    train_dataset = MRIDataset(
        train_ids, os.path.join(path_data, "Training_Set"), 
        slice_size=slice_size, target_size=target_size, n_classes=n_classes, data_augmentation=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create validation dataset
    val_dataset = MRIDataset(
        val_ids, os.path.join(path_data, "Validation_Set"), 
        slice_size=0, target_size=target_size, n_classes=n_classes, data_augmentation=False
    )
    # Validation always processes one volume at a time
    val_loader = DataLoader(
    val_dataset, 
    batch_size=len(val_dataset.samples) // len(val_dataset.volume_map), 
    shuffle=False
    )

    return train_loader, val_loader
