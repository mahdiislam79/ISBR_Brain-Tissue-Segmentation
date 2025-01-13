from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
import torch

class BrainSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.label_filenames = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        image = nib.load(image_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.float32)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        # label = np.expand_dims(label, axis=0)

        return torch.tensor(image), torch.tensor(label)
