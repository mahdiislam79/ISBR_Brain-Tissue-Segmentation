import os
import argparse
import torch
from torch.optim import Adam
from utils import Utils
from configurations import Configuration
from unet import UNet
from metrics import mean_dice_score, weighted_categorical_crossentropy
import wandb

def run_program(config, network_name, training_params):
    # Initialize W&B
    wandb.init(project="brain_tissue_segmentation", name=network_name, config=training_params)

    # Initialize utilities and data loaders
    utils = Utils()
    train_loader, val_loader = config.create_dataloaders()

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=config.n_classes, factor=1).to(device)

    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=training_params['learning_rate'])
    loss_fn = weighted_categorical_crossentropy

    # Callbacks and early stopping
    save_checkpoint, scheduler = utils.all_callbacks(network_name, optimizer, wandb_run=wandb)
    early_stopping = utils.early_stopping(patience=training_params['early_stopping_patience'])
    history = utils.initialize_history()

    # Training loop
    for epoch in range(training_params['epochs']):
        model.train()
        train_loss, train_dice = 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(batch_y, outputs)
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_dice += mean_dice_score(batch_y, outputs).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss, val_dice = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = loss_fn(batch_y, outputs)

                val_loss += loss.item()
                val_dice += mean_dice_score(batch_y, outputs).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Update history and W&B
        utils.update_history(history, train_loss, val_loss, train_dice, val_dice, wandb_run=wandb)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{training_params['epochs']} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")

        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Save checkpoints
        save_checkpoint(model, epoch)

    # Save final model and training plots
    model_path = f"results/{network_name}/Best_Model.pth"
    torch.save(model.state_dict(), model_path)
    wandb.log_artifact(model_path, type="model_checkpoint")
    utils.save_training_plots(history, f"results/{network_name}/training_plots.png", wandb_run=wandb)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Brain Tissue Segmentation Training Script")

    # Add arguments
    parser.add_argument("--data_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--network_name", type=str, default="Unet", help="Name of the network")
    parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Mini-batch size (number of slices per batch)")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of output classes")
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="Patience for early stopping")

    # Parse arguments
    args = parser.parse_args()

    # Parameters for Configuration
    config_params = {
        'path_data': args.data_path,
        'target_size': (256, 256),
        'batch_size': args.batch_size,
        'mini_batch_size': args.mini_batch_size,
        'n_classes': args.n_classes,
    }

    # Training-specific parameters
    training_params = {
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'early_stopping_patience': args.early_stopping_patience,
    }

    # Create results directory
    os.makedirs(f"results/{args.network_name}", exist_ok=True)

    # Initialize Configuration
    config = Configuration(**config_params)

    # Run the training program
    run_program(config, args.network_name, training_params)
