import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Import AIDO.Cell specific packages
from aido_cell.models import CellFoundationModel, CellFoundationConfig


torch.set_float32_matmul_precision("medium")

# Set multiprocessing start method to 'spawn' to avoid fork issues
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


class MockDataset(IterableDataset):

    def __init__(self, batch_size, sleep_time, input_size, output_size, n_batches):
        self.sleep_time = sleep_time
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size

    def __iter__(self):
        # Handle worker info for DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split work among workers
            per_worker = int(np.ceil(self.n_batches / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_batches)
        else:
            iter_start = 0
            iter_end = self.n_batches

        # Create tensors in the worker process to avoid sharing memory
        for _ in range(iter_start, iter_end):
            time.sleep(self.sleep_time)
            x_mock = torch.ones((self.batch_size, self.input_size), dtype=torch.float32)
            y_mock = torch.ones(self.batch_size, dtype=torch.int64)
            yield {
                "counts": x_mock,
                "label": y_mock
            }


class CellFoundationClassifier(pl.LightningModule):
    """AIDO.Cell model with classification head."""

    def __init__(self, model_name, num_classes, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained AIDO.Cell model
        config = CellFoundationConfig.from_pretrained(model_name)
        self.backbone = CellFoundationModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.learning_rate = learning_rate

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        # Get embeddings from backbone
        # Ensure input_ids is float for gene expression values (not int/long)
        # The model will handle precision conversion for mixed precision training
        if not input_ids.is_floating_point():
            input_ids = input_ids.float()

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Mean pooling (weighted by attention mask)
        last_hidden_state = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1)
        if last_hidden_state.dtype == torch.bfloat16:
            mask_expanded = mask_expanded.to(torch.bfloat16)
        else:
            mask_expanded = mask_expanded.float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        pooled = sum_embeddings / sum_mask

        # Classification
        logits = self.classifier(pooled)
        return logits

    def training_step(self, batch, batch_idx):
        counts = batch["counts"]
        labels = batch["label"]

        # Create attention mask (all ones since we're using mock data)
        attention_mask = torch.ones(counts.shape[0], counts.shape[1], device=counts.device)

        # Ensure counts are float type (not long/int64) to match model dtype
        # The model will handle dtype conversion internally for mixed precision
        if counts.dtype != torch.float32:
            counts = counts.float()

        # Forward pass
        logits = self(counts, attention_mask)
        loss = self.criterion(logits, labels)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


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
        num_workers=0,  # Disable multiprocessing in DataLoader to avoid conflicts with DDP
        persistent_workers=False,
    )


def train_model(model_name, num_classes, learning_rate, batch_size, input_dim, n_classes, n_batches, strategy, devices, accumulate_grad_batches=1):
    """
    Train the model across different loading speeds and measure fit time.

    Args:
        model_name: Hugging Face model name
        num_classes: Number of output classes
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        input_dim: Input dimension (number of genes)
        n_classes: Number of classes (for MockDataset compatibility)
        n_batches: Number of batches per epoch
        strategy: Training strategy ('ddp', 'ddp_spawn', etc.)
        devices: Number of GPUs to use or list of device IDs
        accumulate_grad_batches: Number of batches to accumulate gradients

    Returns:
        DataFrame with sleep times, fit times, and loading speeds
    """
    res = {"sleep": [], "fit_time": [], "samples_per_sec": []}

    # Single near-zero sleep to measure pure compute time
    for sleep in [1e-10]:
        # Benchmark loader performance for this sleep time
        loader = create_loader(batch_size, sleep, input_dim, n_classes, n_batches)
        sps, _, _ = benchmark_loader(
            loader,
            batch_size * n_batches,
            batch_size
        )

        # Reinitialize model for each iteration to ensure fair comparison
        model = CellFoundationClassifier(
            model_name=model_name,
            num_classes=num_classes,
            learning_rate=learning_rate
        )

        trainer = pl.Trainer(
            max_steps=200,
            logger=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            strategy=strategy,
            devices=devices,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=1.0,
            precision="bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed" if torch.cuda.is_available() else "32",
        )

        start = time.time()
        trainer.fit(
            model,
            train_dataloaders=create_loader(batch_size, sleep, input_dim, n_classes, n_batches)
        )
        elapsed = time.time() - start

        res["fit_time"].append(elapsed)
        res["sleep"].append(sleep)
        res["samples_per_sec"].append(sps)

        # Clean up resources to prevent leaks
        del model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
        description="Benchmark AIDO.Cell model with varying data loading speeds"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients (default: 1)"
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=2000,
        help="Input dimension (number of genes, default: 2000)"
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
        default="ddp_spawn",
        choices=["ddp", "ddp_spawn", "ddp_notebook", "fsdp"],
        help="Multi-GPU training strategy (default: ddp_spawn)"
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
        default="genbio-ai/AIDO.Cell-3M",
        help="Hugging Face model name (default: genbio-ai/AIDO.Cell-3M)"
    )

    args = parser.parse_args()

    # Check GPU availability
    if not torch.cuda.is_available():
        args.devices = 1
        args.strategy = "auto"  # Use 'auto' for CPU to avoid DDP
    else:
        n_gpus = torch.cuda.device_count()
        if args.devices == -1:
            args.devices = n_gpus
        # Use 'auto' for single GPU to avoid DDP overhead
        if args.devices == 1:
            args.strategy = "auto"

    # Constants for MockDataset
    N_CLASSES = 100  # For classification head

    # Train model across different loading speeds
    results = train_model(
        args.model_name,
        N_CLASSES,
        args.learning_rate,
        args.batch_size,
        args.input_dim,
        N_CLASSES,
        args.n_batches,
        args.strategy,
        args.devices,
        args.accumulate_grad_batches
    )

    # Add metadata to results
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
