import os
import nibabel as nib
import numpy as np
import random
import torch
import torch.nn.functional as F

class Preprocessing:
    def __init__(self, data_aug=True, norm_intensity=True):
        self.data_aug = data_aug
        self.norm_intensity = norm_intensity

    def read_image(self, image_path):
        img = nib.load(image_path)
        img = img.get_fdata()
        return img

    def data_augmentation(self, images, masks):
        augmented_img, augmented_mask = [], []
        for img, mask in zip(images, masks):
            if random.random() > 0.5:
                img = np.flip(img, axis=0)
                mask = np.flip(mask, axis=0)
            if random.random() > 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)
            augmented_img.append(img)
            augmented_mask.append(mask)
        return augmented_img, augmented_mask

    def zscore(self, img):
        lower_percentile = np.percentile(img, 25)
        upper_percentile = np.percentile(img, 75)
        filtered_array = img[(img >= lower_percentile) & (img <= upper_percentile)]
        mean = np.mean(filtered_array)
        std = np.std(filtered_array)
        z_scores = (img - mean) / std
        return z_scores

    def normalize_intensity(self, img):
        img[img > 0] = self.zscore(img[img > 0])
        return img

    def pad_image_center(self, image, target_size=256, axis=0):
        pad_total = target_size - image.shape[axis]
        pad1 = pad_total // 2
        pad2 = pad_total - pad1
        padding = [(0, 0), (0, 0)]
        padding[axis] = (pad1, pad2)
        return np.pad(image, padding, mode='constant', constant_values=0)

    def select_slices(self, img, gt, number_slices):
        all_slices_x, all_slices_y = [], []
        csf_slices_x, csf_slices_y = [], []

        for slice_idx in range(gt.shape[1]):
            slice_y = gt[:, slice_idx, :]
            slice_x = img[:, slice_idx, :]

            if not np.all(slice_y == 0):
                if np.any(slice_y == 1):
                    csf_slices_x.append(slice_x)
                    csf_slices_y.append(slice_y)
                else:
                    all_slices_x.append(slice_x)
                    all_slices_y.append(slice_y)

        if len(csf_slices_y) >= number_slices:
            indices = random.sample(range(len(csf_slices_y)), number_slices)
            return [csf_slices_x[i] for i in indices], [csf_slices_y[i] for i in indices]

        remaining = number_slices - len(csf_slices_y)
        indices = random.sample(range(len(all_slices_y)), remaining)
        csf_slices_x.extend([all_slices_x[i] for i in indices])
        csf_slices_y.extend([all_slices_y[i] for i in indices])

        return csf_slices_x, csf_slices_y

    def to_one_hot(self, y, num_classes):
        """
        Converts a label tensor to one-hot encoding.

        Parameters:
        - y: Tensor of labels with shape (batch, height, width).
        - num_classes: Number of classes for one-hot encoding.

        Returns:
        - One-hot encoded tensor with the correct dimensions: (batch, num_classes, height, width).
        """
        # Ensure y is a PyTorch tensor
        y = torch.tensor(y, dtype=torch.long)

        # Apply one-hot encoding and rearrange to (batch, num_classes, height, width)
        y_one_hot = F.one_hot(y, num_classes=num_classes).permute(0, 3, 1, 2)

        return y_one_hot


    def process_case(self, ID, path, number_slices, num_classes):
        path_img = os.path.join(path, ID, f"{ID}.nii.gz")
        path_gt = os.path.join(path, ID, f"{ID}_seg.nii.gz")

        # Read images
        img = self.read_image(path_img)
        gt = self.read_image(path_gt)

        # Normalize intensities
        if self.norm_intensity:
            img = self.normalize_intensity(img)

        # Select slices with some criteria
        if number_slices > 0:
            img, gt = self.select_slices(img, gt, number_slices)
        else:
            img = np.transpose(img[..., 0], [1, 0, 2])
            gt = np.transpose(gt[..., 0], [1, 0, 2])

        # Data augmentation
        if self.data_aug:
            img, gt = self.data_augmentation(img, gt)

        # Convert to arrays
        img = np.stack(img)  # Shape: (slices, height, width)
        gt = np.stack(gt)  # Shape: (slices, height, width)

        # # Add the channel dimension to img (for grayscale)
        # img = np.expand_dims(img, axis=1)  # Shape: (slices, 1, height, width)

        # One-hot encode the labels
        gt = self.to_one_hot(gt, num_classes)  # Shape: (slices, num_classes, height, width)

        return torch.tensor(img, dtype=torch.float32), gt


