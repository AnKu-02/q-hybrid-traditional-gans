#!/usr/bin/env python3
"""
Entry point for Conditional GAN training.

Usage:
    python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml
    python scripts/train_cgan.py --config configs/cgan_roi_128.yaml

Official Paper:
- "Conditional Generative Adversarial Nets" (Mirza & Osindski, 2014)
  https://arxiv.org/abs/1411.1784
"""

import sys
from pathlib import Path

# Add train module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.train_cgan import train_cgan, load_config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Conditional GAN for NEU-DET defect generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml
  python scripts/train_cgan.py --config configs/cgan_roi_128.yaml

Paper Reference:
  "Conditional Generative Adversarial Nets" (Mirza & Osindski, 2014)
  https://arxiv.org/abs/1411.1784
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    
    args = parser.parse_args()
    config_path = args.config
    
    # Verify config exists
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading config from: {config_path}")
    
    # Load config
    config = load_config(config_path)
    
    # Train
    train_cgan(config)


if __name__ == "__main__":
    main()
