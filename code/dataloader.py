import os
from dataset import MRIDataset
from torch.utils.data import DataLoader

def get_dataloaders(path_data, train_ids, val_ids, minibatch_size=32, target_size=(256, 256), n_classes=1):
    """
    Creates DataLoaders for training and validation datasets.

    Parameters:
    - path_data: Path to the data directory.
    - train_ids: List of training image IDs.
    - val_ids: List of validation image IDs.
    - minibatch_size: Number of slices in a mini-batch.
    - target_size: Desired size of each image.
    - n_classes: Number of segmentation classes.

    Returns:
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    """
    train_dataset = MRIDataset(
        train_ids, os.path.join(path_data, "Training_Set"), 
        minibatch_size=minibatch_size, target_size=target_size, n_classes=n_classes, data_augmentation=True
    )

    val_dataset = MRIDataset(
        val_ids, os.path.join(path_data, "Training_Set"), 
        minibatch_size=minibatch_size, target_size=target_size, n_classes=n_classes, data_augmentation=False
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader
