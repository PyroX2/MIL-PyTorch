# Setup
1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Sync packages and python version
```bash
uv sync
```

# Training on single GPU
```bash
uv run train.py --data-dir path_to_dataset
```

# Training with multiple GPUs
1. Set environment variable with number of GPUs to use during training
```bash
export N_GPU=4  # In this case 4 GPUs will be used
```

2. Run training:
```bash
source .venv/bin/activate   # Activate virtual environment
torchrun --nproc_per_node=$N_GPU train.py --data-dir path_to_dataset
```