import os
import torch
import json
from torchvision.transforms import v2
from image_patcher import ImagePatcher
from dataset import MILDataset
from data_utils import collate_fn
from metrics import BinaryMetricsCalculator, MulticlassMetricsCalculator
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd
from ddp_utils import init_distributed, cleanup_distributed, gather_from_ranks
from data_utils import create_dataloader
from model_utils import build_model
from torch.utils.data import random_split
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
    parser.add_argument("--output-dim", type=int, required=False, default=-1, help="Number of output neurons. Should be equal to n_classes for classification problem or 1 for binary classification.")
    parser.add_argument("--ckpt-path", type=str, required=False, default=None, help="Path to model weights used for evaluation")
    return parser.parse_args()


# Given a model and validation dataloader, evaluate the model performance on validation set
def validate(model, val_dl, criterion, output_dim, is_ddp, rank, world_size, device):
    # Initialize validation dataloader with correct number of classes
    if output_dim == 1:
        metrics_calculator = BinaryMetricsCalculator()
    else:
        metrics_calculator = MulticlassMetricsCalculator(num_classes=output_dim)

    val_loss = 0.0 # Track validation loss
    outputs_list = []
    targets_list = []
    losses_list = []

    if rank == 0:
        iterator = tqdm(val_dl, desc="Validation")
    else:
        iterator = val_dl

    i = 0

    model.eval()
    with torch.no_grad():
        for features, labels, masks, bags_length in iterator:
            # Move data to device
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Model forward pass
            outputs = model(features, masks, bags_length)

            # If binary classification use sigmoid and convert labels to float
            if output_dim == 1:
                outputs = F.sigmoid(outputs)
                labels = labels.to(torch.float32)

            loss = criterion(outputs, labels)

            losses_list.append(loss.item())
            val_loss += loss.item()
            outputs_list.extend(outputs.detach().cpu().tolist())
            targets_list.extend(labels.detach().cpu().tolist())

            if i > 100:
                break
            
            i += 1

    gathered_outputs = gather_from_ranks(outputs_list, is_ddp, world_size)
    gathered_targets = gather_from_ranks(targets_list, is_ddp, world_size)
    # gathered_losses = gather_from_ranks(losses_list, is_ddp, world_size)
    gathered_losses = gather_from_ranks(val_loss, is_ddp, world_size)

    if rank != 0:
        return None
    
    gathered_losses = torch.tensor(gathered_losses).flatten()
    gathered_outputs = torch.tensor(gathered_outputs).flatten(0, 1)
    gathered_targets = torch.tensor(gathered_targets).flatten(0, 1)

    avg_val_loss = gathered_losses.mean() / len(val_dl)

    val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix = metrics_calculator.calculate(gathered_outputs, gathered_targets)
    return avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix


def main():
    # Setup distributed data processing
    is_ddp, local_rank, rank, world_size = init_distributed()

    if rank == 0:
        print(f"DDP initialized: is_ddp={is_ddp}, world_size={world_size}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    args = parse_args()
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

    val_dataset = MILDataset(dataset_path=os.path.join(args.data_dir, "val"), image_patcher=patcher, dirs_with_classes=selected_classes, transform=transform)

    val_dataloader, val_sampler = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, sample_type=None, num_workers=args.num_workers, is_ddp=is_ddp, rank=rank, world_size=world_size)

    n_classes = len(val_dataset.classes)

    if args.output_dim == -1:
        if n_classes == 2:
            output_dim = 1
        else:
            output_dim = n_classes
    else:
        output_dim = args.output_dim

    if args.ckpt_path is not None:
        state_dict = torch.load(args.ckpt_path, weights_only=True, map_location=device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    else:
        state_dict = None

    # Initialize model, loss function, and optimizer
    model = build_model(output_dim=output_dim, att_dim=args.attention_dim, is_ddp=is_ddp, rank=rank, local_rank=local_rank, device=device, state_dict=state_dict)

    criterion = torch.nn.CrossEntropyLoss()

    # Calculate validation metrics
    res = validate(
        model, 
        val_dataloader, 
        criterion,
        output_dim=output_dim,
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size,
        device=device)
    
    if res is not None:
        avg_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc, val_precision, val_recall, confusion_matrix = res

        # Convert metrics to pandas data frames
        results_df = pd.DataFrame([[
            avg_val_loss, 
            val_accuracy, 
            val_f1_score, 
            val_auprc, 
            val_auroc, 
            val_precision, 
            val_recall]], 
            columns=[
                "avg_val_loss", 
                "val_accuracy", 
                "val_f1_score", 
                "val_auprc", 
                "val_auroc", 
                "precision", 
                "recall"])
        confusion_matrix_df = pd.DataFrame(confusion_matrix)
        
        # Save results to csv files
        results_df.to_csv("test_results.csv")
        confusion_matrix_df.to_csv("confusion_matrix.csv")


    # Distributed data processing cleanup
    if is_ddp:
        cleanup_distributed()


if __name__ == "__main__":
    main()


