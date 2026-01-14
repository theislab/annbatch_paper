"""
Visualization script for AIDO.Cell-10M benchmark results.
This script creates plots comparing fit time vs loading speed for AIDO.Cell and other models.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(*csv_files):
    """
    Load benchmark results from multiple CSV files.

    Args:
        *csv_files: Variable number of CSV file paths

    Returns:
        Combined DataFrame with all results
    """
    dfs = []
    for csv_file in csv_files:
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            dfs.append(df)
        else:
            print(f"Warning: File not found: {csv_file}")

    if not dfs:
        raise ValueError("No valid CSV files found!")

    return pd.concat(dfs, ignore_index=True)


def plot_fit_time_vs_loading_speed(results, output_prefix="aido_cell_comparison"):
    """
    Create visualization of fit time vs loading speed.

    Args:
        results: DataFrame with benchmark results
        output_prefix: Prefix for output files
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Create the plot
    fig, ax = plt.subplots()

    # Plot each model
    for model_name in results['model'].unique():
        model_data = results[results['model'] == model_name]

        # Create label with GPU info if available
        if 'n_gpus' in model_data.columns:
            n_gpus = model_data['n_gpus'].iloc[0]
            label = f"{model_name} ({n_gpus} GPU{'s' if n_gpus > 1 else ''})"
        else:
            label = model_name

        ax.plot(
            model_data['samples_per_sec'],
            model_data['fit_time'],
            marker='o',
            linewidth=2,
            markersize=8,
            label=label
        )

    # Set labels and scale
    ax.set_xlabel("Data Loading Speed (samples/sec)", fontsize=12)
    ax.set_ylabel("Fit Time (seconds)", fontsize=12)
    ax.set_title("Training Time vs Data Loading Speed", fontsize=14, fontweight='bold')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Add annotations for interesting points
    for model_name in results['model'].unique():
        model_data = results[results['model'] == model_name]

        # Annotate min and max fit times
        min_idx = model_data['fit_time'].idxmin()
        max_idx = model_data['fit_time'].idxmax()

        for idx in [min_idx, max_idx]:
            row = model_data.loc[idx]
            ax.annotate(
                f'{row["fit_time"]:.1f}s',
                (row['samples_per_sec'], row['fit_time']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                alpha=0.7
            )

    plt.tight_layout()

    # Save plots
    png_file = f"{output_prefix}_fit_time.png"
    svg_file = f"{output_prefix}_fit_time.svg"

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(svg_file, bbox_inches='tight')

    print(f"Saved plots: {png_file}, {svg_file}")
    plt.close()


def plot_speedup_analysis(results, output_prefix="aido_cell_comparison"):
    """
    Create visualization showing speedup from fast vs slow data loading.

    Args:
        results: DataFrame with benchmark results
        output_prefix: Prefix for output files
    """
    # Calculate speedup for each model
    speedup_data = []

    for model_name in results['model'].unique():
        model_data = results[results['model'] == model_name]

        min_time = model_data['fit_time'].min()
        max_time = model_data['fit_time'].max()
        speedup = max_time / min_time

        n_gpus = model_data['n_gpus'].iloc[0] if 'n_gpus' in model_data.columns else 1

        speedup_data.append({
            'model': model_name,
            'speedup': speedup,
            'min_time': min_time,
            'max_time': max_time,
            'n_gpus': n_gpus
        })

    speedup_df = pd.DataFrame(speedup_data)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = sns.color_palette("husl", len(speedup_df))
    bars = ax.bar(range(len(speedup_df)), speedup_df['speedup'], color=colors)

    # Customize plot
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Speedup (max_time / min_time)", fontsize=12)
    ax.set_title("Impact of Data Loading Speed on Training Time", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(speedup_df)))
    ax.set_xticklabels(speedup_df['model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, speedup_df.itertuples())):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}x\n({row.n_gpus} GPU)',
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()

    # Save plot
    png_file = f"{output_prefix}_speedup.png"
    svg_file = f"{output_prefix}_speedup.svg"

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(svg_file, bbox_inches='tight')

    print(f"Saved speedup plots: {png_file}, {svg_file}")
    plt.close()

    return speedup_df


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("Benchmark Results Summary")
    print("="*80)

    for model_name in results['model'].unique():
        model_data = results[results['model'] == model_name]

        print(f"\nModel: {model_name}")
        if 'n_gpus' in model_data.columns:
            print(f"  GPUs: {model_data['n_gpus'].iloc[0]}")
        if 'strategy' in model_data.columns:
            print(f"  Strategy: {model_data['strategy'].iloc[0]}")
        if 'batch_size' in model_data.columns:
            print(f"  Batch Size: {model_data['batch_size'].iloc[0]}")

        min_time = model_data['fit_time'].min()
        max_time = model_data['fit_time'].max()
        speedup = max_time / min_time

        print(f"  Min Fit Time: {min_time:.2f}s (fastest loading)")
        print(f"  Max Fit Time: {max_time:.2f}s (slowest loading)")
        print(f"  Speedup: {speedup:.2f}x")

        # Loading speed range
        min_loading = model_data['samples_per_sec'].min()
        max_loading = model_data['samples_per_sec'].max()
        print(f"  Loading Speed Range: {min_loading:.0f} - {max_loading:.0f} samples/sec")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AIDO.Cell-10M benchmark results"
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="CSV files with benchmark results"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="aido_cell_comparison",
        help="Prefix for output plot files (default: aido_cell_comparison)"
    )

    args = parser.parse_args()

    # Load results
    print("Loading benchmark results...")
    results = load_results(*args.csv_files)

    print(f"Loaded {len(results)} data points from {len(results['model'].unique())} model(s)")

    # Print summary
    print_summary(results)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_fit_time_vs_loading_speed(results, args.output_prefix)
    speedup_df = plot_speedup_analysis(results, args.output_prefix)

    # Save combined results
    combined_csv = f"{args.output_prefix}_combined.csv"
    results.to_csv(combined_csv, index=False)
    print(f"\nSaved combined results to: {combined_csv}")

    # Save speedup summary
    speedup_csv = f"{args.output_prefix}_speedup_summary.csv"
    speedup_df.to_csv(speedup_csv, index=False)
    print(f"Saved speedup summary to: {speedup_csv}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

