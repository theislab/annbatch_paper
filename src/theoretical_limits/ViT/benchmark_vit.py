"""
Benchmark throughput of a torchvision Vision Transformer in samples/sec.

Uses a mock data loader that yields the same pre-allocated tensor on every
iteration so the benchmark is never data-loading bound and measures pure
model forward + backward throughput.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
import torchvision.models as tv_models

VIT_VARIANTS = {
    "vit_b_16": tv_models.vit_b_16,
    "vit_b_32": tv_models.vit_b_32,
    "vit_l_16": tv_models.vit_l_16,
    "vit_l_32": tv_models.vit_l_32,
    "vit_h_14": tv_models.vit_h_14,
}

torch.set_float32_matmul_precision("medium")


class MockImageDataset(IterableDataset):
    """Iterable dataset that yields the same pre-allocated (image, label) batch.

    The tensor is created once in __init__ and reused on every iteration,
    ensuring that dataloading overhead is negligible and the benchmark
    measures pure GPU compute throughput.
    """

    def __init__(self, batch_size: int, n_batches: int, channels: int = 3,
                 height: int = 224, width: int = 224, num_classes: int = 1000):
        self.n_batches = n_batches
        self.x_mock = torch.randn(batch_size, channels, height, width, dtype=torch.float32)
        self.y_mock = torch.randint(0, num_classes, (batch_size,), dtype=torch.int64)

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.x_mock, self.y_mock


class ViTClassifier(pl.LightningModule):
    """Torchvision ViT wrapped in a LightningModule for benchmarking."""

    def __init__(self, variant: str = "vit_b_16", num_classes: int = 1000,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        factory = VIT_VARIANTS[variant]
        self.model = factory(weights=None, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                 weight_decay=0.01)


def create_loader(batch_size: int, n_batches: int, image_size: int = 224,
                   num_classes: int = 1000):
    """Create a DataLoader backed by MockImageDataset."""
    return DataLoader(
        MockImageDataset(batch_size, n_batches, height=image_size,
                         width=image_size, num_classes=num_classes),
        batch_size=None,
        num_workers=0,
        persistent_workers=False,
    )


def benchmark_model(variant: str, batch_size: int, n_batches: int, max_steps: int,
                    num_classes: int, learning_rate: float,
                    image_size: int, devices, strategy: str, precision: str):
    """
    Benchmark ViT training throughput.

    Returns a dict with measured metrics.
    """
    model = ViTClassifier(
        variant=variant,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    trainer = pl.Trainer(
        max_steps=max_steps,
        logger=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        strategy=strategy,
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=precision,
    )

    loader = create_loader(batch_size, n_batches, image_size, num_classes)

    warmup_loader = create_loader(batch_size, n_batches=5, image_size=image_size, num_classes=num_classes)
    warmup_trainer = pl.Trainer(
        max_steps=3,
        logger=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        strategy=strategy,
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=precision,
    )
    warmup_trainer.fit(model, train_dataloaders=warmup_loader)
    del warmup_trainer

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    trainer.fit(model, train_dataloaders=loader)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start

    total_samples = max_steps * batch_size
    samples_per_sec = total_samples / elapsed

    result = {
        "model": variant,
        "batch_size": batch_size,
        "image_size": image_size,
        "max_steps": max_steps,
        "total_samples": total_samples,
        "elapsed_sec": round(elapsed, 3),
        "samples_per_sec": round(samples_per_sec, 2),
        "precision": precision,
        "n_gpus": devices if isinstance(devices, int) else len(devices),
        "strategy": strategy,
    }

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ViT training throughput (samples/sec) with mock data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_b_16",
        choices=list(VIT_VARIANTS.keys()),
        help="ViT variant to benchmark (default: vit_b_16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training (default: 128)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Number of training steps to measure (default: 200)",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=500,
        help="Number of batches available in the mock dataset per epoch "
             "(should be >= max-steps, default: 500)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Height and width of mock input images (default: 224)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of output classes (default: 1000, i.e. ImageNet)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=-1,
        help="Number of GPUs to use (-1 for all available, default: -1)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "ddp", "ddp_spawn", "fsdp"],
        help="Multi-GPU training strategy (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file name (default: <model>_throughput.csv)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.devices = 1
        args.strategy = "auto"
    else:
        n_gpus = torch.cuda.device_count()
        if args.devices == -1:
            args.devices = n_gpus
        if args.devices == 1:
            args.strategy = "auto"

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
    else:
        precision = "32"

    if args.output is None:
        args.output = f"{args.model}_throughput.csv"

    print(f"Running {args.model} benchmark: batch_size={args.batch_size}, "
          f"image_size={args.image_size}x{args.image_size}, "
          f"max_steps={args.max_steps}, devices={args.devices}, "
          f"strategy={args.strategy}, precision={precision}")

    result = benchmark_model(
        variant=args.model,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        max_steps=args.max_steps,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        devices=args.devices,
        strategy=args.strategy,
        precision=precision,
    )

    df = pd.DataFrame([result])
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*50}")
    print(f"  {result['model']} Throughput Benchmark Results")
    print(f"{'='*50}")
    print(f"  Batch size:       {result['batch_size']}")
    print(f"  Image size:       {result['image_size']}x{result['image_size']}")
    print(f"  Steps:            {result['max_steps']}")
    print(f"  Total samples:    {result['total_samples']}")
    print(f"  Elapsed time:     {result['elapsed_sec']:.2f} s")
    print(f"  Throughput:       {result['samples_per_sec']:.2f} samples/sec")
    print(f"  Precision:        {result['precision']}")
    print(f"  GPUs:             {result['n_gpus']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
