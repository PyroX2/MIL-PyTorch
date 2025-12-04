import os
import torch
from torch.utils.data import Subset
from torchvision.transforms import v2
from image_patcher import ImagePatcher
from dataset import MILDataset
from data_utils import collate_fn
from model import AttentionMILModel
from metrics import BinaryMetricsCalculator, MulticlassMetricsCalculator
from argparse import ArgumentParser
from ddp_utils import init_distributed, cleanup_distributed, gather_from_ranks
from data_utils import create_dataloader
from model_utils import build_model
import wandb
from tqdm import tqdm
import json
import torch.nn.functional as F


def parse_args():
    parser = ArgumentParser(description="Train Attention-based MIL Model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--patch-size", type=int, default=224, help="Size of image patches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap size between patches")
    parser.add_argument("--attention-dim", type=int, default=128, help="Dimension of attention layer")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--log-wandb", action="store_true", help="Whether to log training to Weights & Biases")
    parser.add_argument("--class-selection", action="store_true", help="If used classes are defined based on classes.json config file")
    parser.add_argument("--sample-type", type=str, default=None, required=False, help="Method used for balancing the dataset. Can be 'oversample', 'undersample' or None")
    return parser.parse_args()

args = parse_args()

def get_logger(args):
    # Start a new wandb run to track this script.
    wandb_logger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="kubawilk63-politechnika-gda-ska",
        # Set the wandb project where this run will be logged.
        project="MIL-Breast-Cancer",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "dataset": args.data_dir,
            "epochs": args.num_epochs,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "attention_dim": args.attention_dim,
            "device": args.device,
        },
    )
    return wandb_logger


# Given a model and validation dataloader, evaluate the model performance on validation set
def validate(model, val_dl, criterion, n_classes, is_ddp, rank, world_size, device):
    # Initialize validation dataloader with correct number of classes
    metrics_calculator = BinaryMetricsCalculator()

    val_loss = 0.0 # Track validation loss
    outputs_list = []
    targets_list = []
    losses_list = []

    if rank == 0:
        iterator = tqdm(val_dl, desc="Validation")
    else:
        iterator = val_dl

    model.eval()
    with torch.no_grad():
        for features, labels, masks, bags_length in iterator:
            # Move data to device
            features = features.to(device)
            labels = labels.to(device).to(torch.float32)
            masks = masks.to(device)

            # Model forward pass
            outputs = model(features, masks, bags_length)
            outputs = F.sigmoid(outputs)
            loss = criterion(outputs.squeeze(), labels)

            losses_list.append(loss.item())
            val_loss += loss.item()
            outputs_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(labels.detach().cpu().tolist())

    gathered_outputs = gather_from_ranks(outputs_list, is_ddp, world_size)
    gathered_targets = gather_from_ranks(targets_list, is_ddp, world_size)
    gathered_losses = gather_from_ranks(val_loss, is_ddp, world_size)

    if rank != 0:
        return None
    
    gathered_losses = torch.tensor(gathered_losses).flatten()
    gathered_outputs = torch.tensor(gathered_outputs).flatten(0, 1)
    gathered_targets = torch.tensor(gathered_targets).flatten(0, 1)

    avg_val_loss = gathered_losses.mean() / len(val_dl)

    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, _ = metrics_calculator.calculate(gathered_outputs, gathered_targets)
    return avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall


# Train the model
def train(model, train_dl, val_dl, train_sampler, criterion, optimizer, device, num_epochs, n_classes, is_ddp, rank, world_size, logger=None):
    # Initialize variables to track best model
    best_val_loss = float('inf')
    
    metrics_calculator = BinaryMetricsCalculator()
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs} started")
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0 # Track training epoch loss
        outputs_list = []
        targets_list = []

        if rank == 0:
            iterator = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        else:
            iterator = train_dl

        model.train()
        for features, labels, masks, bags_length in iterator:
            optimizer.zero_grad() # Zero the gradients

            # Move data to device
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device).to(torch.float32)

            # Model and criterion forward pass
            outputs = model(features, masks, bags_length)
            outputs = F.sigmoid(outputs)
            loss = criterion(outputs, labels)

            # Model optimization step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            outputs_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(labels.detach().cpu().tolist())

        # Calculate train metrics
        avg_train_loss = epoch_loss / len(train_dl)
        train_accuracy, train_f1_score, train_auprc, train_auroc, train_precision, train_recall, _ = metrics_calculator.calculate(outputs_list, targets_list)

        # Calculate validation metrics
        res = validate(
            model, 
            val_dl, 
            criterion,
            n_classes=n_classes,
            is_ddp=is_ddp,
            rank=rank,
            world_size=world_size,
            device=device)
        
        if res is not None:
            avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall = res

            # Print epoch summary
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
            # Logging to wandb
            if logger is not None:
                logger.log({
                    "train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                    "train_f1_score": train_f1_score,
                    "train_auprc": train_auprc,
                    "train_auroc": train_auroc,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "val_f1_score": val_f1_score,
                    "val_auprc": val_auprc,
                    "val_auroc": val_auroc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_attention_mil_model.pth")

    print("Model training complete and saved.")
    torch.save(model.state_dict(), "attention_mil_model.pth")
    
    if logger is not None:
        logger.log_model(path="attention_mil_model.pth", name="final_attention_mil_model")


def main():

    # Setup distributed data processing
    is_ddp, local_rank, rank, world_size = init_distributed()

    if rank == 0:
        print(f"DDP initialized: is_ddp={is_ddp}, world_size={world_size}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    if args.log_wandb and rank == 0 and is_ddp:     # If distributed, only log from rank 0
        wandb_logger = get_logger(args)
        wandb_logger.log_model(path="model.py", name="attention_mil_model")
    elif args.log_wandb and not is_ddp:    # Non-distributed logging
        wandb_logger = get_logger(args)
        wandb_logger.log_model(path="model.py", name="attention_mil_model")
    else:
        wandb_logger = None

    device = args.device

    # Define image transformations
    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
        ])

    # Create patcher used for splitting images into patches
    patcher = ImagePatcher(patch_size=args.patch_size, overlap=args.overlap)

    # Select subset of classes
    if args.class_selection:
        with open("config/classes.json", "r") as f:
            selected_classes = json.load(f)
    else:
        selected_classes = None

    # Create dataset and dataloader
    train_dataset = MILDataset(dataset_path=os.path.join(args.data_dir, "train"), image_patcher=patcher, dirs_with_classes=selected_classes, transform=transform)
    train_dataloader, train_sampler = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, sample_type=args.sample_type, num_workers=args.num_workers, is_ddp=is_ddp, rank=rank, world_size=world_size)

    val_dataset = MILDataset(dataset_path=os.path.join(args.data_dir, "val"), image_patcher=patcher, dirs_with_classes=selected_classes, transform=transform)
    val_dataloader, val_sampler = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, sample_type=args.sample_type, num_workers=args.num_workers, is_ddp=is_ddp, rank=rank, world_size=world_size)

    if wandb_logger is not None:
        wandb_logger.config["classes"] = train_dataset.dirs_with_classes

    n_classes = len(train_dataset.classes)

    # Initialize model, loss function, and optimizer
    model = build_model(output_dim=1, att_dim=args.attention_dim, is_ddp=is_ddp, rank=rank, local_rank=local_rank, device=device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    train(
        model, 
        train_dataloader, 
        val_dataloader, 
        train_sampler, 
        criterion, 
        optimizer, 
        device, 
        num_epochs=args.num_epochs,
        n_classes=n_classes,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size, 
        logger=wandb_logger)

    # Distributed data processing cleanup
    if is_ddp:
        cleanup_distributed()


if __name__ == "__main__":
    main()