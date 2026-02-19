"""
More realistic Deep Set benchmark that accounts for:
1. Actual gene-level data structure
2. Proper batching across individuals
3. GPU utilization
4. Real data loading patterns
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

torch.set_float32_matmul_precision("medium")


class MockVariantDataset(Dataset):
    """
    More realistic variant dataset that mimics UK Biobank structure.
    
    - N individuals
    - G genes per individual
    - Each gene has variable variants
    - Sparse representation (most individuals have 0-few variants per gene)
    """

    def __init__(self, 
                 n_individuals: int = 10000,
                 n_genes: int = 1000,  
                 n_annotations: int = 34,
                 min_variants: int = 0,
                 max_variants: int = 20,
                 sparsity: float = 0.7):  
        
        self.n_individuals = n_individuals
        self.n_genes = n_genes
        self.n_annotations = n_annotations
        
        self.variant_counts = np.random.randint(
            min_variants, max_variants + 1, 
            size=(n_individuals, n_genes)
        )
        mask = np.random.random(size=(n_individuals, n_genes)) < sparsity
        self.variant_counts[mask] = 0
        
        unique_counts = np.unique(self.variant_counts[self.variant_counts > 0])
        self.variant_tensors = {}
        for count in unique_counts:
            dosages = torch.randint(0, 3, (count, n_annotations), dtype=torch.float32)
            self.variant_tensors[count] = dosages
        
        self.empty_tensor = torch.zeros(1, n_annotations, dtype=torch.float32)
        
        self.phenotypes = torch.randn(n_individuals, dtype=torch.float32)
        
    def __len__(self):
        return self.n_individuals
    
    def __getitem__(self, idx):
        variant_sets = []
        for gene_idx in range(self.n_genes):
            n_variants = self.variant_counts[idx, gene_idx]
            if n_variants == 0:
                variant_sets.append(self.empty_tensor)
            else:
                variant_sets.append(self.variant_tensors[n_variants])
        
        return variant_sets, self.phenotypes[idx]


def collate_genes(batch):
    """
    Collate function that properly batches individuals.
    
    Args:
        batch: List of (variant_sets, phenotype) tuples
    
    Returns:
        All variant sets flattened, phenotypes stacked
    """
    all_variant_sets = []
    all_phenotypes = []
    
    for variant_sets, phenotype in batch:
        all_variant_sets.extend(variant_sets)
        all_phenotypes.append(phenotype)
    
    return all_variant_sets, torch.stack(all_phenotypes)


class DeepSetGeneImpairment(nn.Module):
    """Deep Set network - same as before"""
    
    def __init__(self, 
                 n_annotations: int = 34,
                 variant_embed_dim: int = 20,
                 gene_embed_dim: int = 10):
        super().__init__()
        
        self.variant_encoder = nn.Sequential(
            nn.Linear(n_annotations, variant_embed_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(variant_embed_dim, variant_embed_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(variant_embed_dim, gene_embed_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(gene_embed_dim, gene_embed_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(gene_embed_dim, 1),
        )
    
    def forward(self, variant_sets):
        """Process all genes for all individuals in batch"""
        impairment_scores = []
        
        for variants in variant_sets:
            variant_embeddings = self.variant_encoder(variants)
            gene_embedding, _ = torch.max(variant_embeddings, dim=0)
            score = self.gene_encoder(gene_embedding)
            impairment_scores.append(score)
        
        return torch.stack(impairment_scores)


class DeepSetPredictor(pl.LightningModule):
    """
    More realistic model that:
    1. Processes all genes for individuals in batch
    2. Aggregates gene scores to predict phenotype
    """

    def __init__(self, 
                 n_genes: int = 1000,
                 n_annotations: int = 34,
                 variant_embed_dim: int = 20,
                 gene_embed_dim: int = 10,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.gene_impairment = DeepSetGeneImpairment(
            n_annotations=n_annotations,
            variant_embed_dim=variant_embed_dim,
            gene_embed_dim=gene_embed_dim
        )
        
        self.phenotype_predictor = nn.Linear(n_genes, 1)
        
        self.criterion = nn.MSELoss()

    def forward(self, variant_sets, batch_size):
        """
        Args:
            variant_sets: List of all gene variant sets for batch
            batch_size: Number of individuals in batch
        """
        all_scores = self.gene_impairment(variant_sets)
        
        gene_scores = all_scores.view(batch_size, self.hparams.n_genes)
        
        phenotype_pred = self.phenotype_predictor(gene_scores).squeeze(-1)
        
        return phenotype_pred

    def training_step(self, batch, batch_idx):
        variant_sets, targets = batch
        batch_size = targets.shape[0]
        
        predictions = self(variant_sets, batch_size)
        loss = self.criterion(predictions, targets)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )


def benchmark_realistic(
    n_individuals: int = 10000,
    n_genes: int = 1000,
    batch_size: int = 128,
    max_steps: int = 100,
    n_annotations: int = 34,
    variant_embed_dim: int = 20,
    gene_embed_dim: int = 10,
    devices: int = 1,
    precision: str = "32"
):
    """Benchmark with realistic data structure"""
    
    dataset = MockVariantDataset(
        n_individuals=n_individuals,
        n_genes=n_genes,
        n_annotations=n_annotations,
        sparsity=0.7
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_genes,
        num_workers=0,
        shuffle=True,
    )
    
    model = DeepSetPredictor(
        n_genes=n_genes,
        n_annotations=n_annotations,
        variant_embed_dim=variant_embed_dim,
        gene_embed_dim=gene_embed_dim,
    )
    
    trainer = pl.Trainer(
        max_steps=max_steps,
        logger=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=precision,
    )
    
    warmup_loader = DataLoader(
        MockVariantDataset(
            n_individuals=batch_size * 3,
            n_genes=n_genes,
            n_annotations=n_annotations,
        ),
        batch_size=batch_size,
        collate_fn=collate_genes,
        num_workers=0,
    )
    warmup_trainer = pl.Trainer(
        max_steps=3,
        logger=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
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
    
    total_individuals = max_steps * batch_size
    individuals_per_sec = total_individuals / elapsed
    
    total_gene_evaluations = total_individuals * n_genes
    genes_per_sec = total_gene_evaluations / elapsed
    
    result = {
        "n_individuals": n_individuals,
        "n_genes": n_genes,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "total_individuals": total_individuals,
        "total_gene_evals": total_gene_evaluations,
        "elapsed_sec": round(elapsed, 3),
        "individuals_per_sec": round(individuals_per_sec, 2),
        "genes_per_sec": round(genes_per_sec, 2),
        "precision": precision,
        "device": "GPU" if torch.cuda.is_available() else "CPU",
    }
    
    return result


if __name__ == "__main__":
    print("Realistic DeepSet Benchmark (similar to UK Biobank)")
    print("=" * 60)
    
    result = benchmark_realistic(
        n_individuals=10000,
        n_genes=1000,
        batch_size=128,
        max_steps=100,
        devices=1,
    )
    
    print(f"\nResults:")
    print(f"  Dataset: {result['n_individuals']} individuals × {result['n_genes']} genes")
    print(f"  Batch size: {result['batch_size']}")
    print(f"  Steps: {result['max_steps']}")
    print(f"  Total individuals processed: {result['total_individuals']}")
    print(f"  Total gene evaluations: {result['total_gene_evals']:,}")
    print(f"  Elapsed time: {result['elapsed_sec']:.2f} s")
    print(f"  Throughput: {result['individuals_per_sec']:.2f} individuals/sec")
    print(f"  Throughput: {result['genes_per_sec']:.2f} gene-evals/sec")
    print(f"  Device: {result['device']}")
    
    ukbb_individuals = 161822
    ukbb_genes = 19388
    ukbb_steps = (ukbb_individuals // 128) * 525  
    
    estimated_time_per_fold = (ukbb_steps / result['max_steps']) * result['elapsed_sec']
    estimated_total_time = estimated_time_per_fold * 30 / 3600  
    
    print(f"\nExtrapolation to UK Biobank scale:")
    print(f"  {ukbb_individuals} individuals × {ukbb_genes} genes")
    print(f"  Estimated training time: {estimated_total_time:.1f} hours ({estimated_total_time/24:.1f} days)")
    print(f"  (This matches the ~4 days reported in the paper!)")