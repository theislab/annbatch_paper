#!/usr/bin/env python3
"""
Test script to verify AIDO.Cell benchmark installation and setup.
Run this before executing the full benchmark to catch any issues early.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - not installed")
        return False


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA availability...", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            print(f"✓ {n_gpus} GPU(s) available")
            for i in range(n_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
            return True
        else:
            print("✗ CUDA not available (will run on CPU)")
            return False
    except ImportError:
        print("✗ Cannot import torch")
        return False


def check_nvidia_smi():
    """Check if nvidia-smi is available."""
    print("\nChecking nvidia-smi...", end=" ")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ Available")
            return True
        else:
            print("✗ Failed")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ Not found")
        return False


def check_huggingface_auth():
    """Check Hugging Face authentication."""
    print("\nChecking Hugging Face authentication...", end=" ")
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Authenticated")
            return True
        else:
            print("⚠ Not authenticated (may be required for some models)")
            print("  Run: huggingface-cli login")
            return None  # Warning, not error
    except ImportError:
        print("✗ huggingface-hub not installed")
        return False


def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...", end=" ")
    try:
        import shutil
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        if free_gb > 50:
            print(f"✓ {free_gb:.1f} GB available")
            return True
        else:
            print(f"⚠ Only {free_gb:.1f} GB available (recommend 50+ GB for model)")
            return None  # Warning
    except Exception as e:
        print(f"✗ Could not check ({e})")
        return False


def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")
    required_files = [
        "benchmark_aido_cell.py",
        "visualize_aido_cell_results.py",
        "requirements_aido_cell.txt"
    ]

    parent_files = [
        "../mock_dataset.py"
    ]

    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - not found")
            all_exist = False

    for file in parent_files:
        path = Path(file)
        if path.exists():
            print(f"✓ {file} (parent directory)")
        else:
            print(f"✗ {file} - not found")
            all_exist = False

    return all_exist


def test_mock_dataset():
    """Test if MockDataset can be imported and used."""
    print("\nTesting MockDataset...", end=" ")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mock_dataset import MockDataset
        import torch

        # Create a small test dataset
        dataset = MockDataset(
            batch_size=32,
            sleep_time=0.001,
            input_size=100,
            output_size=10,
            n_batches=2
        )

        # Test iteration
        for x, y in dataset:
            assert x.shape == (32, 100)
            assert y.shape == (32,)
            break

        print("✓ Working correctly")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_gpu_simple():
    """Test simple GPU operation."""
    print("\nTesting GPU operation...", end=" ")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            x = torch.randn(100, 100, device=device)
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print("✓ GPU operations working")
            return True
        else:
            print("⚠ Skipped (no CUDA)")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("="*70)
    print("AIDO.Cell Benchmark - Installation Test")
    print("="*70)

    results = {}

    # Check Python version
    results['python'] = check_python_version()

    # Check required packages
    print("\nChecking required packages...")
    packages = [
        ('torch', 'torch'),
        ('pytorch-lightning', 'pytorch_lightning'),
        ('transformers', 'transformers'),
        ('accelerate', 'accelerate'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]

    for pkg_name, import_name in packages:
        results[pkg_name] = check_package(pkg_name, import_name)

    # Check CUDA
    results['cuda'] = check_cuda()

    # Check nvidia-smi
    results['nvidia-smi'] = check_nvidia_smi()

    # Check Hugging Face auth
    results['hf-auth'] = check_huggingface_auth()

    # Check disk space
    results['disk'] = check_disk_space()

    # Check files
    results['files'] = check_files()

    # Test MockDataset
    results['mock_dataset'] = test_mock_dataset()

    # Test GPU operations
    results['gpu_ops'] = test_gpu_simple()

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    warnings = sum(1 for v in results.values() if v is None)

    print(f"Passed: {passed}")
    print(f"Warnings: {warnings}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\n⚠ Some checks failed. Please install missing packages:")
        print("   pip install -r requirements_aido_cell.txt")
        return False
    elif warnings > 0:
        print("\n⚠ Some warnings detected. Review the output above.")
        print("   You can still run the benchmark, but some features may not work.")
        return True
    else:
        print("\n✓ All checks passed! You're ready to run the benchmark.")
        print("\nNext steps:")
        print("   1. python benchmark_aido_cell.py --batch-size 4096 --n-batches 3")
        print("   2. python benchmark_aido_cell.py  # Full benchmark")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

