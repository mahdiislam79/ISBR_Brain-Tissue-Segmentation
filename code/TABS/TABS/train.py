import argparse
import os
import random
import logging
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim

import torch.distributed as dist
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import nibabel as nib

from dataset import BrainSegmentationDataset

from Models.TABS_Model import TABS

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

assert torch.cuda.is_available(), "CUDA is not available. Check runtime and configuration."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

parser = argparse.ArgumentParser()
parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
# Root directory
parser.add_argument('--root', default='', type=str)
# learning rate
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--end_epoch', default=20, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--gpu_available', default='0,1,2', type=str)

args = parser.parse_args()

def main_worker():

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)  # discouraged

    model = TABS(img_dim=192, patch_dim=8, img_ch=1, output_ch=3, embedding_dim=512, num_heads=8, num_layers=4)

    # Load pretrained weights for fine-tuning
    pretrained_weights_path = '/content/drive/MyDrive/MAIA_Work/Semester_3/MISA/MISA_Project/best_model_TABS.pth'
    checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)  # Allow fine-tuning

    model.cuda(args.gpu)

    print('Model Built and Pretrained Weights Loaded!')

    # Using adam optimizer (amsgrad variant) with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

    # MSE loss for this task (regression). Using reduction value of sum because we want to specify the number of voxels to divide by (only in the brain map)
    criterion = nn.MSELoss(reduction='sum')
    criterion = criterion.cuda(args.gpu)

    # *************************************************************************
    # Place train and validation datasets/dataloaders here
    # *************************************************************************

    # Initialize training and validation datasets and loaders
    train_dataset = BrainSegmentationDataset(
        image_dir='/content/drive/MyDrive/MAIA_Work/Semester_3/MISA/MISA_Project/ProcessedDataV4/Training_Set/Image',
        label_dir='/content/drive/MyDrive/MAIA_Work/Semester_3/MISA/MISA_Project/ProcessedDataV4/Training_Set/Label_ProbMaps'
    )
    val_dataset = BrainSegmentationDataset(
        image_dir='/content/drive/MyDrive/MAIA_Work/Semester_3/MISA/MISA_Project/ProcessedDataV4/Validation_Set/Image',
        label_dir='/content/drive/MyDrive/MAIA_Work/Semester_3/MISA/MISA_Project/ProcessedDataV4/Validation_Set/Label_ProbMaps'
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


    start_time = time.time()

    # Enable gradient calculation for training
    torch.set_grad_enabled(True)

    # Declare lists to keep track of training and val losses over the epochs
    train_global_losses = []
    val_global_losses = []
    best_epoch = 0

    print('Start to train!')

    # Main training/validation loop
    for epoch in range(args.start_epoch, args.end_epoch):

        # Declare lists to keep track of losses and metrics within the epoch
        train_epoch_losses = []
        val_epoch_losses = []
        val_epoch_pcorr = []
        val_epoch_psnr = []
        val_epoch_dice = []
        start_epoch = time.time()

        model.train()

        # Loop through train dataloader here.
        for mri_images, targets in train_loader:

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            mri_images = mri_images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

            loss, isolated_images, stacked_brain_map  = get_loss(model, criterion, mri_images, targets, 'train')

            train_epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Transition to val mode
        model.eval()

        with torch.no_grad():

            # Loop through validation dataloader here.
            for mri_images, tagets in val_loader:

                mri_images = mri_images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

                loss, isolated_images, stacked_brain_map  = get_loss(model, criterion, mri_images, targets, 'val')

                val_epoch_losses.append(loss.item())

                # for j in range(0,len(isolated_images)):
                #     cur_pcorr = overall_metrics(isolated_images[j], targets[j], stacked_brain_map[j])
                #     val_epoch_pcorr.append(cur_pcorr)

                for j in range(len(isolated_images)):
                    try:
                        cur_pcorr, cur_dice = overall_metrics(isolated_images[j], targets[j], stacked_brain_map[j])
                        val_epoch_pcorr.append(cur_pcorr)
                        val_epoch_dice.extend(cur_dice)

                    except IndexError as e:
                        print(f"IndexError at index {j}: {e}")
                        print(f"Isolated images shape: {isolated_images.shape}")
                        print(f"Targets shape: {targets.shape}")
                        print(f"Stacked brain map shape: {stacked_brain_map.shape}")
                        break

        end_epoch = time.time()

        # Average train and val loss over every MRI scan in the epoch. Save to global losses which tracks across epochs
        train_net_loss = sum(train_epoch_losses) / len(train_epoch_losses)
        val_net_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        train_global_losses.append(train_net_loss)
        val_global_losses.append(val_net_loss)
        pcorr = sum(val_epoch_pcorr) / len(val_epoch_pcorr)

        # Average Dice scores for each tissue type
        num_tissue_types = 3  # Assuming 3 tissues: CSF, GM, WM
        average_dice_scores = [0] * num_tissue_types
        count_dice_scores = [0] * num_tissue_types

        # Aggregate Dice scores
        for i, dice in enumerate(val_epoch_dice):
            tissue_idx = i % num_tissue_types
            average_dice_scores[tissue_idx] += dice
            count_dice_scores[tissue_idx] += 1

        average_dice_scores = [
            avg / count if count > 0 else 0 for avg, count in zip(average_dice_scores, count_dice_scores)
        ]

        print(f"Epoch: {epoch} | Train Loss: {train_net_loss:.4f} | Val Loss: {val_net_loss:.4f} | Pearson: {pcorr:.4f} | Dice: {average_dice_scores}")

        checkpoint_dir = args.root
        # Save the model if it reaches a new min validation loss
        if val_global_losses[-1] == min(val_global_losses):
            print('saving model at the end of epoch ' + str(epoch))
            best_epoch = epoch
            file_name = os.path.join(checkpoint_dir, 'TABS_model_epoch_{}_val_loss_{}.pth'.format(epoch, val_global_losses[-1]))
            # Only save model at higher epochs
            if epoch > 150:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    },
                    file_name)

    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    print('The total training time is {:.2f} hours'.format(total_time))

    print('----------------------------------The training process finished!-----------------------------------')

    # log_name = os.path.join(args.root, args.protocol, 'loss_log_restransunet.txt')
    log_name = os.path.join(args.root, 'loss_log_TABS.txt')

    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Loss (%s) ================\n' % now)
        log_file.write('best_epoch: ' + str(best_epoch) + '\n')
        log_file.write('train_losses: ')
        log_file.write('%s\n' % train_global_losses)
        log_file.write('val_losses: ')
        log_file.write('%s\n' % val_global_losses)
        log_file.write('train_time: ' + str(total_time))

    learning_curve(best_epoch, train_global_losses, val_global_losses)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Input the best epoch, lists of global (across epochs) train and val losses. Plot learning curve
def learning_curve(best_epoch, train_global_losses, val_global_losses):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Epochs')
    ax1.set_xticks(np.arange(0, int(len(train_global_losses) + 1), 10))

    ax1.set_ylabel('Loss')
    ax1.plot(train_global_losses, '-r', label='Training loss', markersize=3)
    ax1.plot(val_global_losses, '-b', label='Validation loss', markersize=3)
    ax1.axvline(best_epoch, color='m', lw=4, alpha=0.5, label='Best epoch')
    ax1.legend(loc='upper left')
    save_name = 'Learning_Curve_TABS' + '.png'
    plt.savefig(os.path.join(args.root, save_name))

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch / max_epoch), power), 8)

# # Calculate pearson correlation and psnr only between the voxels of the brain map (do by total brain not tissue type during training)
# def overall_metrics(isolated_image, target, stacked_brain_map):
#     # Flatten the GT, isolated output, and brain mask
#     GT_flattened = torch.flatten(target)
#     iso_flattened = torch.flatten(isolated_image)
#     mask_flattened = torch.flatten(stacked_brain_map)

#     # Only save the part of the flattened GT/output that corresponds to nonzero values of the brain mask
#     GT_flattened = GT_flattened[mask_flattened.nonzero(as_tuple=True)]
#     iso_flattened = iso_flattened[mask_flattened.nonzero(as_tuple=True)]

#     iso_flattened = iso_flattened.cpu().detach().numpy()
#     GT_flattened = GT_flattened.cpu().detach().numpy()

#     pearson = np.corrcoef(iso_flattened, GT_flattened)[0][1]

#     return pearson

def overall_metrics(isolated_image, target, stacked_brain_map):
    # Flatten the GT, isolated output, and brain mask
    GT_flattened = torch.flatten(target)
    iso_flattened = torch.flatten(isolated_image)
    mask_flattened = torch.flatten(stacked_brain_map)

    # Only keep brain voxels
    GT_flattened = GT_flattened[mask_flattened.nonzero(as_tuple=True)]
    iso_flattened = iso_flattened[mask_flattened.nonzero(as_tuple=True)]

    # Convert to numpy arrays
    iso_flattened = iso_flattened.cpu().detach().numpy()
    GT_flattened = GT_flattened.cpu().detach().numpy()

    # Pearson correlation
    pearson = np.corrcoef(iso_flattened, GT_flattened)[0][1]

    # Dice coefficient calculation
    dice_scores = []
    for i in range(isolated_image.shape[0]):  # Loop through each tissue type
        pred_binary = isolated_image[i]
        gt_binary = (target[i] > 0.5).float()

        intersection = torch.sum(pred_binary * gt_binary)
        dice = (2.0 * intersection) / (torch.sum(pred_binary) + torch.sum(gt_binary) + 1e-6)
        dice_scores.append(dice.item())

    return pearson, dice_scores

# Given the model, criterion, input, and GT, this function calculates the loss and returns the isolated output (stripped of background) and brain map
def get_loss(model, criterion, mri_images, targets, mode):

    if mode == 'val':
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Gen model outputs
    output = model(mri_images)

    # Construct binary brain map to consider loss only within there
    input_squeezed = torch.squeeze(mri_images, dim=1)
    brain_map = (input_squeezed > -1).float()
    stacked_brain_map = torch.stack([brain_map, brain_map, brain_map], dim=1)

    # Zero out the background of the segmentation output
    isolated_images = torch.mul(stacked_brain_map, output)

    # Calculate loss over just the brain map
    loss = criterion(isolated_images, targets)
    num_brain_voxels = stacked_brain_map.sum()
    loss = loss / num_brain_voxels

    return loss, isolated_images, stacked_brain_map

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_available
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
