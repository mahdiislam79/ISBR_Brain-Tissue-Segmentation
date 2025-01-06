import torch
import torch.nn as nn
import torch.nn.functional as F

def single_dice_score(y_true, y_pred, nclass, smooth=1.0):
    """Computes the Dice coefficient for a specific class."""
    y_true_f = y_true[:, nclass, ...].contiguous().view(-1)
    y_pred_f = y_pred[:, nclass, ...].contiguous().view(-1)
    
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def mean_dice_score(y_true, y_pred):
    """Computes the mean Dice coefficient across CSF, GM, and WM classes."""
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()
    y_true = y_true.float()

    dice_csf = single_dice_score(y_true, y_pred, 1)
    dice_gm = single_dice_score(y_true, y_pred, 2)
    dice_wm = single_dice_score(y_true, y_pred, 3)

    return (dice_csf + dice_gm + dice_wm) / 3.0

def csf(y_true, y_pred, smooth=1.0):
    """Computes Dice score for CSF."""
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()
    y_true = y_true.float()
    return single_dice_score(y_true, y_pred, 1, smooth)

def gm(y_true, y_pred, smooth=1.0):
    """Computes Dice score for GM."""
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()
    y_true = y_true.float()
    return single_dice_score(y_true, y_pred, 2, smooth)

def wm(y_true, y_pred, smooth=1.0):
    """Computes Dice score for WM."""
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()
    y_true = y_true.float()
    return single_dice_score(y_true, y_pred, 3, smooth)

def bg(y_true, y_pred, smooth=1.0):
    """Computes Dice score for Background."""
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = F.one_hot(y_pred, num_classes=4).permute(0, 3, 1, 2).float()
    y_true = y_true.float()
    return single_dice_score(y_true, y_pred, 0, smooth)

def weighted_categorical_crossentropy(y_true, y_pred, weights=[1, 10, 3, 3]):
    """Implements weighted categorical cross-entropy loss."""
    # Scale predictions to ensure class probabilities sum to 1
    y_pred = F.softmax(y_pred, dim=-1)

    # Clip predictions to prevent NaNs and Infs
    y_pred = torch.clamp(y_pred, min=torch.finfo(y_pred.dtype).eps, max=1.0 - torch.finfo(y_pred.dtype).eps)

    # Apply weights
    weights = torch.tensor(weights, dtype=torch.float32, device=y_pred.device).view(1, -1, 1, 1)
    loss = y_true * torch.log(y_pred) * weights

    # Compute the weighted loss
    loss = -torch.sum(loss) / weights[y_true.argmax(dim=1)].sum()
    return torch.mean(loss)



