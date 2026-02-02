#!/usr/bin/env python3
"""
StyleGAN2-ADA Training Script
=============================

Usage:
    python scripts/train_stylegan2_ada.py --config configs/stylegan2_ada_baseline_128.yaml

References:
    - Paper: "Training Generative Adversarial Networks with Limited Data"
      https://arxiv.org/abs/2006.06676
    - Official Implementation: https://github.com/NVlabs/stylegan2-ada-pytorch
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train.train_stylegan2_ada import StyleGAN2ADAConfig, train_stylegan2_ada
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Train StyleGAN2-ADA on NEU defect dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default config
    python scripts/train_stylegan2_ada.py --config configs/stylegan2_ada_baseline_128.yaml
    
    # Override specific parameters
    python scripts/train_stylegan2_ada.py \\
        --config configs/stylegan2_ada_baseline_128.yaml \\
        --epochs 50 \\
        --batch-size 64 \\
        --device cuda

References:
    - StyleGAN2-ADA Paper: https://arxiv.org/abs/2006.06676
    - Official Repository: https://github.com/NVlabs/stylegan2-ada-pytorch
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Override device"
    )
    parser.add_argument(
        "--lr-g",
        type=float,
        default=None,
        help="Override generator learning rate"
    )
    parser.add_argument(
        "--lr-d",
        type=float,
        default=None,
        help="Override discriminator learning rate"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"📋 Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override with command-line arguments
    if args.epochs is not None:
        config_dict['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
    if args.device is not None:
        config_dict['device'] = args.device
    if args.lr_g is not None:
        config_dict['learning_rate_g'] = args.lr_g
    if args.lr_d is not None:
        config_dict['learning_rate_d'] = args.lr_d
    
    # Create config object
    config = StyleGAN2ADAConfig(**config_dict)
    
    print("\n" + "=" * 70)
    print("StyleGAN2-ADA Configuration")
    print("=" * 70)
    print(f"Dataset: {config.metadata_path}")
    print(f"Image Size: {config.img_size}x{config.img_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Device: {config.device}")
    print(f"Learning Rates: G={config.learning_rate_g}, D={config.learning_rate_d}")
    print(f"Z Dimension: {config.z_dim}")
    print(f"W Dimension: {config.w_dim}")
    print("=" * 70 + "\n")
    
    # Train
    train_stylegan2_ada(config)


if __name__ == "__main__":
    main()
