import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import numpy as np
import os
from matplotlib.colors import ListedColormap
import pandas as pd

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def all_callbacks(network_name, optimizer, patience_early_stopping=20, patience_reduce_lr=10, factor=0.1, min_lr=1e-6, wandb_run=None):
        """
        Sets up callbacks for PyTorch equivalent functionality.
        - Saves model checkpoints.
        - Reduces learning rate on plateau.
        - Early stopping functionality is managed externally in training logic.
        - Logs metrics and artifacts to W&B if `wandb_run` is provided.
        """
        # Model checkpoint directory
        os.makedirs(f"results/{network_name}/1", exist_ok=True)
        checkpoint_path = f"results/{network_name}/1"

        # Model checkpoint saver
        def save_checkpoint(model, epoch):
            save_path = os.path.join(checkpoint_path, f"epoch-{epoch:02d}.pt")
            torch.save(model.state_dict(), save_path)
            if wandb_run:
                wandb_run.log_artifact(save_path, type="model_checkpoint")

        # Reduce learning rate on plateau (handled by PyTorch scheduler)
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,  # Optimizer required for ReduceLROnPlateau
            mode='min',  # Reduce LR when monitored metric stops decreasing
            patience=patience_reduce_lr,  # Wait this many epochs before reducing LR
            factor=factor,  # Reduce LR by this factor
            min_lr=min_lr,  # Minimum LR to avoid over-reducing
            verbose=True  # Print LR reduction info
        )

        return save_checkpoint, scheduler

    @staticmethod
    def save_training_plots(history, file_path, wandb_run=None):
        """
        Saves training and validation loss and Dice coefficient plots.

        Parameters:
        - history: A dictionary containing metrics (train_loss, val_loss, val_dice_csf, val_dice_gm, val_dice_wm, val_dice).
        - file_path: File path to save the plot.
        - wandb_run: W&B run object for logging the plot (optional).
        """
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        val_dice_csf = history['val_dice_csf']
        val_dice_gm = history['val_dice_gm']
        val_dice_wm = history['val_dice_wm']
        val_dice = history['val_dice']
        train_dice = history['train_dice']
        epochs = len(train_loss)

        # Prepare data for boxplot of Dice coefficients
        dice_data = pd.DataFrame({
            'Epoch': list(range(1, epochs + 1)) * 3,
            'Dice Coefficient': val_dice_csf + val_dice_gm + val_dice_wm,
            'Type': ['CSF'] * epochs + ['GM'] * epochs + ['WM'] * epochs
        })

        plt.figure(figsize=(16, 10))

        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        sns.lineplot(x=range(1, epochs + 1), y=train_loss, label='Training Loss')
        sns.lineplot(x=range(1, epochs + 1), y=val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Boxplot of Dice coefficients
        plt.subplot(2, 2, 2)
        sns.boxplot(x='Type', y='Dice Coefficient', data=dice_data, palette='Set2')
        plt.xlabel('Dice Type')
        plt.ylabel('Dice Coefficient')
        plt.title('Boxplot of Dice Coefficients (CSF, GM, WM)')

        # Plot validation Dice coefficients for GM, WM, and CSF
        plt.subplot(2, 2, 3)
        sns.lineplot(x=range(1, epochs + 1), y=val_dice_csf, label='CSF Dice Coefficient')
        sns.lineplot(x=range(1, epochs + 1), y=val_dice_gm, label='GM Dice Coefficient')
        sns.lineplot(x=range(1, epochs + 1), y=val_dice_wm, label='WM Dice Coefficient')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Coefficient')
        plt.title('Validation Dice Coefficients (CSF, GM, WM)')
        plt.legend()

        # Plot mean validation Dice coefficient
        plt.subplot(2, 2, 4)
        sns.lineplot(x=range(1, epochs + 1), y=train_dice, label='Mean Training Dice Coefficient')
        sns.lineplot(x=range(1, epochs + 1), y=val_dice, label='Mean Validation Dice Coefficient')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Coefficient')
        plt.title('Mean Training and Validation Dice Coefficient')
        plt.legend()

        # Save the plots
        plt.tight_layout()
        plt.savefig(file_path)

        if wandb_run:
            wandb_run.log({"Training Plots": wandb.Image(file_path)})


    # @staticmethod
    # def save_visualization_comparison(pred, gt, output_dir, num_samples=5):
    #     """
    #     Saves visual comparisons between predictions and ground truth.

    #     Parameters:
    #     - pred: Tensor of predicted labels (softmaxed/logits), shape (batch, n_classes, height, width).
    #     - gt: Tensor of ground truth labels (one-hot encoded), shape (batch, n_classes, height, width).
    #     - output_dir: Directory where the visualizations will be saved.
    #     - num_samples: Number of samples to visualize (randomly selected).
    #     """
    #     os.makedirs(output_dir, exist_ok=True)

    #     # Map class indices to colors for visualization (change as needed)
    #     class_colors = np.array([
    #         [0, 0, 0],         # Background (Black)
    #         [255, 0, 0],       # CSF (Red)
    #         [0, 255, 0],       # GM (Green)
    #         [0, 0, 255]        # WM (Blue)
    #     ], dtype=np.uint8)

    #     # Ensure pred and gt are tensors
    #     if isinstance(pred, np.ndarray):
    #         pred = torch.tensor(pred)
    #     if isinstance(gt, np.ndarray):
    #         gt = torch.tensor(gt)

    #     # Convert predictions and ground truth to class indices
    #     pred_classes = torch.argmax(pred, dim=1).cpu().numpy()  # Shape: (batch, height, width)
    #     gt_classes = torch.argmax(gt, dim=1).cpu().numpy()      # Shape: (batch, height, width)

    #     # Select random samples to visualize
    #     total_samples = pred_classes.shape[0]
    #     selected_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)

    #     for idx in selected_indices:
    #         pred_image = class_colors[pred_classes[idx]]  # Shape: (height, width, 3)
    #         gt_image = class_colors[gt_classes[idx]]      # Shape: (height, width, 3)

    #         # Plot comparison
    #         plt.figure(figsize=(10, 5))
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(pred_image.astype(np.uint8))
    #         plt.title("Prediction")
    #         plt.axis("off")

    #         plt.subplot(1, 2, 2)
    #         plt.imshow(gt_image.astype(np.uint8))
    #         plt.title("Ground Truth")
    #         plt.axis("off")

    #         # Save visualization
    #         save_path = os.path.join(output_dir, f"comparison_{idx}.png")
    #         plt.savefig(save_path)
    #         plt.close()

    @staticmethod
    def early_stopping(patience):
        """
        Implements early stopping functionality.

        Parameters:
        - patience: Number of epochs with no improvement to wait before stopping training.

        Returns:
        - A function to check if training should stop.
        """
        best_loss = float('inf')
        epochs_no_improve = 0

        def should_stop(current_loss):
            nonlocal best_loss, epochs_no_improve
            if current_loss < best_loss:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            return epochs_no_improve >= patience

        return should_stop

    @staticmethod
    def initialize_history():
        """
        Initializes a dictionary to track training and validation metrics.

        Returns:
        - A dictionary with empty lists for loss and Dice coefficients.
        """
        return {
            'train_loss': [],
            'val_loss': [],
            'val_dice_csf': [],
            'val_dice_gm': [],
            'val_dice_wm': [],
            'train_dice': [],
            'val_dice': []
        }

    @staticmethod
    def update_history(history, train_loss, val_loss, val_dice_csf, val_dice_gm, val_dice_wm, train_dice, val_dice, wandb_run=None):
        """
        Updates the history dictionary with metrics for the current epoch and logs to W&B.

        Parameters:
        - history: The history dictionary to update.
        - train_loss: Training loss for the epoch.
        - val_loss: Validation loss for the epoch.
        - val_dice: Validation Dice coefficient for the epoch.
        - val_dice_csf: Validation Dice coefficient for CSF.
        - val_dice_gm: Validation Dice coefficient for GM.
        - val_dice_wm: Validation Dice coefficient for WM.
        - wandb_run: W&B run object for logging metrics (optional).
        """
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice_csf'].append(val_dice_csf)
        history['val_dice_gm'].append(val_dice_gm)
        history['val_dice_wm'].append(val_dice_wm)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)

        if wandb_run:
            wandb_run.log({
                'Train Loss': train_loss,
                'Validation Loss': val_loss,
                'Training Dice Coefficient': train_dice,
                'Validation Dice Coefficient': val_dice,
                'CSF Dice Coefficient': val_dice_csf,
                'GM Dice Coefficient': val_dice_gm,
                'WM Dice Coefficient': val_dice_wm,
                
            })


