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
from time import gmtime, strftime


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
    parser.add_argument("--output-dim", type=int, required=False, default=-1, help="Number of output neurons. Should be equal to n_classes for classification problem or 1 for binary classification.")
    parser.add_argument("--log-name", type=str, required=False, default=strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
    parser.add_argument("--avg-method", type=str, required=False, default="micro", help="Averaging method used by metrics calculator. One of [micro, macro, None]")
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
            "avg_method": args.avg_method
        },
    )
    return wandb_logger


def log_metric(logger: wandb.Run, metric: torch.Tensor, metric_name: str):
    assert isinstance(metric, torch.Tensor), f"Expected metric to be torch.Tensor, found {metric_name} of type {type(metric)}"

    if metric.ndim != 0:
        # Log separate metric for each class
        for class_id in range(metric.size()[0]):
            logger.log({f"{metric_name}_{class_id}": metric[class_id]})
    else:
        logger.log({metric_name: metric})


# Given a model and validation dataloader, evaluate the model performance on validation set
def validate(model, val_dl, criterion, output_dim, is_ddp, rank, world_size, device):
    # Initialize validation dataloader with correct number of classes
    if output_dim == 1:
        metrics_calculator = BinaryMetricsCalculator()
    else:
        metrics_calculator = MulticlassMetricsCalculator(num_classes=output_dim, avg_method=args.avg_method)

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
            labels = labels.to(device)
            masks = masks.to(device)

            # Model forward pass
            outputs = model(features, masks, bags_length)

            # If binary classification use sigmoid and transform labels to float
            if output_dim == 1:
                outputs = F.sigmoid(outputs)
                labels = labels.to(torch.float32)
    
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

    avg_val_loss = torch.tensor(gathered_losses.mean() / len(val_dl))

    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, _ = metrics_calculator.calculate(gathered_outputs, gathered_targets)
    return avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall


# Train the model
def train(model, train_dl, val_dl, train_sampler, criterion, optimizer, device, num_epochs, output_dim, is_ddp, rank, world_size, log_name, logger=None):
    # Initialize variables to track best model
    best_val_loss = float('inf')
    
    # Use correct metrics calculator for classification problem
    if output_dim == 1:
        metrics_calculator = BinaryMetricsCalculator()
    else:
        metrics_calculator = MulticlassMetricsCalculator(num_classes=output_dim, avg_method=args.avg_method)
    
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
            labels = labels.to(device)

            # Model and criterion forward pass
            outputs = model(features, masks, bags_length)

            if output_dim == 1:
                outputs = F.sigmoid(outputs)
                labels = labels.to(torch.float32)

            loss = criterion(outputs, labels)

            # Model optimization step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            outputs_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(labels.detach().cpu().tolist())

        # Calculate train metrics
        avg_train_loss = torch.tensor(epoch_loss / len(train_dl))
        train_accuracy, train_f1_score, train_auprc, train_auroc, train_precision, train_recall, _ = metrics_calculator.calculate(outputs_list, targets_list)

        # Calculate validation metrics
        res = validate(
            model, 
            val_dl, 
            criterion,
            output_dim=output_dim,
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
                log_metric(logger, avg_train_loss, "train_loss")
                log_metric(logger, train_accuracy, "train_accuracy")
                log_metric(logger, train_f1_score, "train_f1_score")
                log_metric(logger, train_auprc, "train_auprc")
                log_metric(logger, train_auroc, "train_auroc")
                log_metric(logger, train_precision, "train_precision")
                log_metric(logger, train_recall, "train_recall")
                log_metric(logger, avg_val_loss, "val_loss")
                log_metric(logger, val_accuracy, "val_accuracy")
                log_metric(logger, val_f1_score, "val_f1_score")
                log_metric(logger, val_auprc, "val_auprc")
                log_metric(logger, val_auroc, "val_auroc")
                log_metric(logger, val_precision, "val_precision")
                log_metric(logger, val_recall, "val_recall")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{log_name}_best_attention_mil_model.pth")

    print("Model training complete and saved.")
    torch.save(model.state_dict(), f"{log_name}_attention_mil_model.pth")
    
    if logger is not None:
        logger.log_model(path=f"{log_name}_attention_mil_model.pth", name="final_attention_mil_model")


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
    val_dataloader, val_sampler = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, sample_type=None, num_workers=args.num_workers, is_ddp=is_ddp, rank=rank, world_size=world_size)


    n_classes = len(train_dataset.classes)

    if args.output_dim == -1:
        if n_classes == 2:
            output_dim = 1
        else:
            output_dim = n_classes
    else:
        output_dim = args.output_dim

    # Initialize model, loss function, and optimizer
    model = build_model(output_dim=output_dim, att_dim=args.attention_dim, is_ddp=is_ddp, rank=rank, local_rank=local_rank, device=device)

    # Use correct criterion for binary/multiclass classification problem
    if output_dim == 1:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Log additional params to wandb logger
    if wandb_logger is not None:
        wandb_logger.config["classes"] = train_dataset.dirs_with_classes
        wandb_logger.config["output_dim"] = output_dim
        wandb_logger.config["optimizer"] = type(optimizer)
        wandb_logger.config["is_ddp"] = is_ddp
        wandb_logger.config["world_size"] = world_size

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
        output_dim=output_dim,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size, 
        logger=wandb_logger,
        log_name=args.log_name)

    # Distributed data processing cleanup
    if is_ddp:
        cleanup_distributed()


if __name__ == "__main__":
    main()