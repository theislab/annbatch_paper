"""
Benchmark throughput of genomic prediction models: Standard FNN vs GenNet-like sparse architecture.

Uses mock genomic data (SNP variants) to measure pure model forward + backward throughput
without data loading bottlenecks.
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

torch.set_float32_matmul_precision("medium")


class MockGenomicDataset(IterableDataset):
    """Iterable dataset that yields the same pre-allocated genomic batch.
    
    Simulates SNP data where each sample has n_snps variants.
    The tensor is created once in __init__ and reused on every iteration.
    """

    def __init__(self, batch_size: int, n_batches: int, n_snps: int = 100000):
        self.n_batches = n_batches

        self.x_mock = torch.randn(batch_size, n_snps, dtype=torch.float32)

        self.y_mock = torch.randint(0, 2, (batch_size,), dtype=torch.int64)

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.x_mock, self.y_mock


class StandardFNN(nn.Module):
    """
    Standard fully-connected neural network baseline.
    
    Architecture: Input → Hidden → Output
    This is the typical LASSO/Ridge regression neural network equivalent.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 2):
        super().__init__()
        
        self.network = nn.Sequential(

            nn.BatchNorm1d(input_size, affine=False),
            
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size, affine=False),
            
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class GenNetLikeSparse(nn.Module):
    """
    GenNet-like architecture with sparse connectivity.
    
    Architecture: SNPs → Genes → Output
    
    Uses sparse matrix multiplication to connect SNPs only to their
    corresponding genes, mimicking biological gene annotations.
    """

    def __init__(self, n_snps: int, n_genes: int, num_classes: int = 2,
                 sparsity: float = 0.01):
        super().__init__()
        self.n_snps = n_snps
        self.n_genes = n_genes
        self.sparsity = sparsity
        
        self.register_buffer('snp_to_gene_mask', self._create_sparse_mask())
        
        self.snp_to_gene = nn.Linear(n_snps, n_genes, bias=False)
        self._apply_sparse_mask()
        
        self.gene_norm = nn.BatchNorm1d(n_genes, affine=False)
        self.gene_activation = nn.Tanh()
        
        self.gene_to_output = nn.Linear(n_genes, num_classes)

    def _create_sparse_mask(self):
        """
        Create sparse connectivity mask between SNPs and genes.
        
        Simulates biological reality where each gene is influenced by
        only a small subset of SNPs (those within/near the gene).
        """
        mask = torch.zeros(self.n_genes, self.n_snps)
        
        snps_per_gene = int(self.n_snps * self.sparsity)
        
        for gene_idx in range(self.n_genes):
            snp_indices = torch.randperm(self.n_snps)[:snps_per_gene]
            mask[gene_idx, snp_indices] = 1.0
        
        return mask

    def _apply_sparse_mask(self):
        """Apply sparsity mask to the weight matrix."""
        with torch.no_grad():
            self.snp_to_gene.weight.data *= self.snp_to_gene_mask

    def forward(self, x):
        gene_features = self.snp_to_gene(x)
        
        gene_features = self.gene_activation(gene_features)
        gene_features = self.gene_norm(gene_features)
        
        output = self.gene_to_output(gene_features)
        
        return output


class GenNetLikeMultiLayer(nn.Module):
    """
    GenNet-like architecture with multiple hierarchical layers.
    
    Architecture: SNPs → Genes → Pathways → Output
    
    This better matches the GenNet paper's multi-layer architecture.
    """

    def __init__(self, n_snps: int, n_genes: int, n_pathways: int = 200,
                 num_classes: int = 2, snp_to_gene_sparsity: float = 0.01,
                 gene_to_pathway_sparsity: float = 0.1):
        super().__init__()
        self.n_snps = n_snps
        self.n_genes = n_genes
        self.n_pathways = n_pathways
        
        self.snp_to_gene = nn.Linear(n_snps, n_genes, bias=False)
        self.register_buffer('mask1', self._create_sparse_mask(n_genes, n_snps, snp_to_gene_sparsity))
        self._apply_mask(self.snp_to_gene, self.mask1)
        
        self.gene_activation = nn.Tanh()
        self.gene_norm = nn.BatchNorm1d(n_genes, affine=False)
        
        self.gene_to_pathway = nn.Linear(n_genes, n_pathways, bias=False)
        self.register_buffer('mask2', self._create_sparse_mask(n_pathways, n_genes, gene_to_pathway_sparsity))
        self._apply_mask(self.gene_to_pathway, self.mask2)
        
        self.pathway_activation = nn.Tanh()
        self.pathway_norm = nn.BatchNorm1d(n_pathways, affine=False)
        
        self.pathway_to_output = nn.Linear(n_pathways, num_classes)

    def _create_sparse_mask(self, n_out, n_in, sparsity):
        """Create sparse connectivity mask."""
        mask = torch.zeros(n_out, n_in)
        connections_per_output = int(n_in * sparsity)
        
        for out_idx in range(n_out):
            in_indices = torch.randperm(n_in)[:connections_per_output]
            mask[out_idx, in_indices] = 1.0
        
        return mask

    def _apply_mask(self, layer, mask):
        """Apply sparsity mask to layer weights."""
        with torch.no_grad():
            layer.weight.data *= mask

    def forward(self, x):
        x = self.snp_to_gene(x)
        x = self.gene_activation(x)
        x = self.gene_norm(x)

        x = self.gene_to_pathway(x)
        x = self.pathway_activation(x)
        x = self.pathway_norm(x)
        
        x = self.pathway_to_output(x)
        
        return x


class GenomicClassifier(pl.LightningModule):
    """Lightning wrapper for genomic models."""

    def __init__(self, model_type: str = "fnn", n_snps: int = 100000,
                 n_genes: int = 20000, n_pathways: int = 200,
                 hidden_size: int = 128, num_classes: int = 2,
                 learning_rate: float = 1e-3, sparsity: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        
        if model_type == "fnn":
            self.model = StandardFNN(
                input_size=n_snps,
                hidden_size=hidden_size,
                num_classes=num_classes
            )
        elif model_type == "gennet_sparse":
            self.model = GenNetLikeSparse(
                n_snps=n_snps,
                n_genes=n_genes,
                num_classes=num_classes,
                sparsity=sparsity
            )
        elif model_type == "gennet_multilayer":
            self.model = GenNetLikeMultiLayer(
                n_snps=n_snps,
                n_genes=n_genes,
                n_pathways=n_pathways,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def create_loader(batch_size: int, n_batches: int, n_snps: int = 100000):
    """Create a DataLoader backed by MockGenomicDataset."""
    return DataLoader(
        MockGenomicDataset(batch_size, n_batches, n_snps=n_snps),
        batch_size=None,
        num_workers=0,
        persistent_workers=False,
    )


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_model(model_type: str, batch_size: int, n_batches: int, 
                    max_steps: int, n_snps: int, n_genes: int, 
                    n_pathways: int, hidden_size: int, num_classes: int,
                    learning_rate: float, sparsity: float,
                    devices, strategy: str, precision: str):
    """
    Benchmark genomic model training throughput.
    
    Returns a dict with measured metrics.
    """
    model = GenomicClassifier(
        model_type=model_type,
        n_snps=n_snps,
        n_genes=n_genes,
        n_pathways=n_pathways,
        hidden_size=hidden_size,
        num_classes=num_classes,
        learning_rate=learning_rate,
        sparsity=sparsity
    )
    
    total_params, trainable_params = count_parameters(model)

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

    loader = create_loader(batch_size, n_batches, n_snps)

    warmup_loader = create_loader(batch_size, n_batches=5, n_snps=n_snps)
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
        "model_type": model_type,
        "batch_size": batch_size,
        "n_snps": n_snps,
        "n_genes": n_genes if model_type != "fnn" else "N/A",
        "n_pathways": n_pathways if model_type == "gennet_multilayer" else "N/A",
        "hidden_size": hidden_size if model_type == "fnn" else "N/A",
        "sparsity": sparsity if "gennet" in model_type else "N/A",
        "total_params": total_params,
        "trainable_params": trainable_params,
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
        description="Benchmark genomic models: Standard FNN vs GenNet-like sparse architectures"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="fnn",
        choices=["fnn", "gennet_sparse", "gennet_multilayer"],
        help="Model architecture to benchmark"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of training steps to measure (default: 100)",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=500,
        help="Number of batches in mock dataset (default: 500)",
    )
    parser.add_argument(
        "--n-snps",
        type=int,
        default=100000,
        help="Number of SNPs (input dimension, default: 100000)",
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=20000,
        help="Number of genes (GenNet intermediate layer, default: 20000)",
    )
    parser.add_argument(
        "--n-pathways",
        type=int,
        default=200,
        help="Number of pathways (GenNet multilayer only, default: 200)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer size for FNN (default: 128)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.01,
        help="Sparsity level for GenNet (fraction of connections, default: 0.01)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes (default: 2 for binary classification)",
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
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "ddp", "ddp_spawn"],
        help="Multi-GPU training strategy (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="genomic_benchmark.csv",
        help="Output CSV file name (default: genomic_benchmark.csv)",
    )
    parser.add_argument(
        "--run-all",
        action="store_false", 
        help="Run all model types and save comparison",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.devices = 1
        args.strategy = "auto"
        precision = "32"
    else:
        if args.devices == 1:
            args.strategy = "auto"
        
        if torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"

    if args.run_all:
        model_types = ["fnn", "gennet_sparse", "gennet_multilayer"]
        results = []
        
        for model_type in model_types:
            print(f"\n{'='*70}")
            print(f"Benchmarking: {model_type}")
            print(f"{'='*70}")
            
            result = benchmark_model(
                model_type=model_type,
                batch_size=args.batch_size,
                n_batches=args.n_batches,
                max_steps=args.max_steps,
                n_snps=args.n_snps,
                n_genes=args.n_genes,
                n_pathways=args.n_pathways,
                hidden_size=args.hidden_size,
                num_classes=args.num_classes,
                learning_rate=args.learning_rate,
                sparsity=args.sparsity,
                devices=args.devices,
                strategy=args.strategy,
                precision=precision,
            )
            results.append(result)
        
        df = pd.DataFrame(results)
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")
        
        print("\nThroughput Comparison:")
        print(df[["model_type", "total_params", "samples_per_sec", "elapsed_sec"]].to_string(index=False))
        
    else:
        print(f"\n{'='*70}")
        print(f"Running {args.model_type} benchmark:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  SNPs: {args.n_snps:,}")
        if args.model_type != "fnn":
            print(f"  Genes: {args.n_genes:,}")
            if args.model_type == "gennet_multilayer":
                print(f"  Pathways: {args.n_pathways}")
            print(f"  Sparsity: {args.sparsity}")
        else:
            print(f"  Hidden size: {args.hidden_size}")
        print(f"  Steps: {args.max_steps}")
        print(f"  Devices: {args.devices}, Strategy: {args.strategy}, Precision: {precision}")
        print(f"{'='*70}\n")

        result = benchmark_model(
            model_type=args.model_type,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            max_steps=args.max_steps,
            n_snps=args.n_snps,
            n_genes=args.n_genes,
            n_pathways=args.n_pathways,
            hidden_size=args.hidden_size,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            sparsity=args.sparsity,
            devices=args.devices,
            strategy=args.strategy,
            precision=precision,
        )

        df = pd.DataFrame([result])
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        print(f"\n{'='*70}")
        print(f"  {result['model_type'].upper()} Throughput Benchmark Results")
        print(f"{'='*70}")
        print(f"  Model:            {result['model_type']}")
        print(f"  Total params:     {result['total_params']:,}")
        print(f"  Trainable params: {result['trainable_params']:,}")
        print(f"  Batch size:       {result['batch_size']}")
        print(f"  Input dimension:  {result['n_snps']:,} SNPs")
        print(f"  Steps:            {result['max_steps']}")
        print(f"  Total samples:    {result['total_samples']:,}")
        print(f"  Elapsed time:     {result['elapsed_sec']:.2f} s")
        print(f"  Throughput:       {result['samples_per_sec']:.2f} samples/sec")
        print(f"  Precision:        {result['precision']}")
        print(f"  GPUs:             {result['n_gpus']}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()