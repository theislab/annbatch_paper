"""
Benchmark script for AIDO.Cell-10M model with multi-GPU training.
This script tests fit time vs data loading speed using the GenBio-AI AIDO.Cell-10M model.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig

# Add parent directory to path to import mock_dataset
sys.path.insert(0, str(Path(__file__).parent.parent))
from mock_dataset import MockDataset


torch.set_float32_matmul_precision("high")


class AIDOCellModel(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the AIDO.Cell-10M model from Hugging Face.
    Adapted for single-cell gene expression data training.
    """

    def __init__(
        self,
        model_name: str = "genbio-ai/AIDO.Cell-10M",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            trust_remote_code=True
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    def forward(self, x):
        """Forward pass through the model."""
        # AIDO.Cell models typically expect gene expression input
        # Adjust based on the actual model API
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch

        # Forward pass - adapt based on model's expected input format
        outputs = self.forward(x)

        # Compute loss - this depends on the model's output format
        # For a generic approach, we'll use a simple reconstruction loss
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            # Simple MSE loss for demonstration
            loss = torch.nn.functional.mse_loss(hidden.mean(dim=1), x)
        elif isinstance(outputs, torch.Tensor):
            loss = torch.nn.functional.mse_loss(outputs, x)
        else:
            # If outputs is a dict with 'loss' key (common for HF models)
            loss = outputs.get('loss', None)
            if loss is None:
                raise ValueError("Could not extract loss from model outputs")

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        outputs = self.forward(x)

        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            loss = torch.nn.functional.mse_loss(hidden.mean(dim=1), x)
        elif isinstance(outputs, torch.Tensor):
            loss = torch.nn.functional.mse_loss(outputs, x)
        else:
            loss = outputs.get('loss', None)
            if loss is None:
                raise ValueError("Could not extract loss from model outputs")

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer with warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Optional: Add learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.warmup_steps / self.trainer.estimated_stepping_batches,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


def create_loader(batch_size, sleep_time, input_dim, n_classes, n_batches):
    """Create a DataLoader with MockDataset."""
    return DataLoader(
        MockDataset(
            batch_size,
            sleep_time,
            input_dim,
            n_classes,
            n_batches,
        ),
        batch_size=None,
    )


def train_model(model, batch_size, input_dim, n_classes, n_batches, strategy, devices):
    """
    Train the model across different loading speeds and measure fit time.

    Args:
        model: The PyTorch Lightning model to train
        batch_size: Batch size for training
        input_dim: Input dimension (number of genes)
        n_classes: Number of classes (not used for AIDO.Cell)
        n_batches: Number of batches per epoch
        strategy: Training strategy ('ddp', 'ddp_spawn', etc.)
        devices: Number of GPUs to use or list of device IDs

    Returns:
        DataFrame with sleep times and fit times
    """
    res = {"sleep": [], "fit_time": []}

    # Test across different loading speeds
    # From fast to slow: 4 * 10^(-3) to 4 * 10^1
    for sleep in 4. * np.logspace(-3, 1, 8):
        trainer = pl.Trainer(
            max_steps=20,
            logger=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            strategy=strategy,
            devices=devices,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )

        start = time.time()
        trainer.fit(
            model,
            train_dataloaders=create_loader(batch_size, sleep, input_dim, n_classes, n_batches)
        )
        elapsed = time.time() - start

        res["fit_time"].append(elapsed)
        res["sleep"].append(sleep)


        # Reinitialize model for next iteration to ensure fair comparison
        model = type(model)(**model.hparams)

    return pd.DataFrame(res)


def benchmark_loader(loader, n_samples, batch_size):
    """
    Benchmark the data loader performance.

    Args:
        loader: DataLoader to benchmark
        n_samples: Total number of samples to load
        batch_size: Batch size

    Returns:
        Tuple of (samples_per_sec, time_per_sample, batch_times)
    """
    num_iter = n_samples // batch_size
    loader_iter = iter(loader)

    start_time = time.time()
    batch_times = []
    batch_time = time.time()

    for i, _batch in enumerate(loader_iter):
        batch_times.append(time.time() - batch_time)
        batch_time = time.time()
        if i >= num_iter:
            break

    execution_time = time.time() - start_time
    time_per_sample = (1e6 * execution_time) / (num_iter * batch_size)
    samples_per_sec = num_iter * batch_size / execution_time


    return samples_per_sec, time_per_sample, batch_times


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AIDO.Cell-10M model with varying data loading speeds"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32768,
        help="Batch size for training (default: 32768)"
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=20000,
        help="Input dimension (number of genes, default: 20000)"
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=7,
        help="Number of batches per epoch (default: 7)"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=-1,
        help="Number of GPUs to use (-1 for all available, default: -1)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "ddp_spawn", "ddp_notebook", "fsdp"],
        help="Multi-GPU training strategy (default: ddp)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="aido_cell_fit_time_vs_loading_speed.csv",
        help="Output CSV file name (default: aido_cell_fit_time_vs_loading_speed.csv)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="genbio-ai/AIDO.Cell-10M",
        help="Hugging Face model name (default: genbio-ai/AIDO.Cell-10M)"
    )

    args = parser.parse_args()

    # Check GPU availability
    if not torch.cuda.is_available():
        args.devices = 1
        args.strategy = "auto"
    else:
        n_gpus = torch.cuda.device_count()
        if args.devices == -1:
            args.devices = n_gpus

    # Constants
    N_CLASSES = 100  # Not really used for AIDO.Cell but needed for MockDataset

    # Initialize model
    model = AIDOCellModel(
        model_name=args.model_name,
        learning_rate=args.learning_rate
    )

    # Train model across different loading speeds
    results = train_model(
        model,
        args.batch_size,
        args.input_dim,
        N_CLASSES,
        args.n_batches,
        args.strategy,
        args.devices
    )

    # Benchmark loader performance for each sleep time
    samples_per_sec = []
    for sleep_time in results.sleep:
        loader = create_loader(
            args.batch_size,
            sleep_time,
            args.input_dim,
            N_CLASSES,
            args.n_batches
        )
        sps, _, _ = benchmark_loader(
            loader,
            args.batch_size * args.n_batches,
            args.batch_size
        )
        samples_per_sec.append(sps)

    results["samples_per_sec"] = samples_per_sec
    results["model"] = args.model_name
    results["batch_size"] = args.batch_size
    results["n_gpus"] = args.devices if isinstance(args.devices, int) else len(args.devices)
    results["strategy"] = args.strategy

    # Save results
    output_path = Path(args.output)
    results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Print summary
    min_fit_time = results.fit_time.min()
    max_fit_time = results.fit_time.max()
    speedup = max_fit_time / min_fit_time
    print(f"Min fit time: {min_fit_time:.2f}s | Max fit time: {max_fit_time:.2f}s | Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()

