import os
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

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
        - history: A dictionary containing metrics (`train_loss`, `val_loss`, `train_dice`, `val_dice`).
        - file_path: File path to save the plot.
        - wandb_run: W&B run object for logging the plot (optional).
        """
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        train_dice = history['train_dice']
        val_dice = history['val_dice']

        epochs = len(train_loss)

        plt.figure(figsize=(12, 5))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_loss, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation Dice coefficient
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_dice, label='Training Dice Coefficient')
        plt.plot(range(1, epochs + 1), val_dice, label='Validation Dice Coefficient')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Coefficient')
        plt.title('Training and Validation Dice Coefficient')
        plt.legend()

        # Save the plots
        plt.tight_layout()
        plt.savefig(file_path)

        if wandb_run:
            wandb_run.log({"Training Plots": wandb.Image(file_path)})

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
            'train_dice': [],
            'val_dice': []
        }

    @staticmethod
    def update_history(history, train_loss, val_loss, train_dice, val_dice, wandb_run=None):
        """
        Updates the history dictionary with metrics for the current epoch and logs to W&B.

        Parameters:
        - history: The history dictionary to update.
        - train_loss: Training loss for the epoch.
        - val_loss: Validation loss for the epoch.
        - train_dice: Training Dice coefficient for the epoch.
        - val_dice: Validation Dice coefficient for the epoch.
        - wandb_run: W&B run object for logging metrics (optional).
        """
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)

        if wandb_run:
            wandb_run.log({
                'Train Loss': train_loss,
                'Validation Loss': val_loss,
                'Train Dice Coefficient': train_dice,
                'Validation Dice Coefficient': val_dice
            })

# # Example Usage
# if __name__ == "__main__":
#     # Initialize W&B
#     wandb.init(project="segmentation_project", name="unet_training")

#     # Example model and optimizer
#     model = torch.nn.Linear(10, 2)  # Replace with your model
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # Initialize history and callbacks
#     history = Utils.initialize_history()
#     save_checkpoint, scheduler = Utils.all_callbacks("unet", wandb_run=wandb)

#     # Training loop
#     early_stopping_checker = Utils.early_stopping(patience=5)
#     for epoch in range(50):  # Example number of epochs
#         train_loss, val_loss = 0.4, 0.35  # Replace with actual training/validation loss
#         train_dice, val_dice = 0.7, 0.75  # Replace with actual metrics

#         # Log metrics and update history
#         Utils.update_history(history, train_loss, val_loss, train_dice, val_dice, wandb_run=wandb)

#         # Save checkpoints
#         save_checkpoint(model, epoch)

#         # Check early stopping
#         if early_stopping_checker(val_loss):
#             print(f"Stopping early at epoch {epoch + 1}")
#             break

#     # Save final training plots
#     Utils.save_training_plots(history, "training_plots.png", wandb_run=wandb)
