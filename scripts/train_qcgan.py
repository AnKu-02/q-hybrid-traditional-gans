"""
Quantum Conditional GAN - Training Entry Point
===============================================

Usage:
    python scripts/train_qcgan.py --config configs/qcgan_baseline_128.yaml
    
    With parameter overrides:
    python scripts/train_qcgan.py \
        --config configs/qcgan_baseline_128.yaml \
        --epochs 50 \
        --qubits 12 \
        --batch-size 8
"""

import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass
import sys

# Add train directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "train"))

from train_qcgan import QuantumCGANConfig, train_qcgan


def main():
    parser = argparse.ArgumentParser(
        description="Train Quantum Conditional GAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with config file
  python scripts/train_qcgan.py --config configs/qcgan_baseline_128.yaml
  
  # Override parameters
  python scripts/train_qcgan.py --config configs/qcgan_baseline_128.yaml \\
    --epochs 50 --qubits 10 --batch-size 8
  
  # High-depth quantum circuit
  python scripts/train_qcgan.py --config configs/qcgan_baseline_128.yaml \\
    --qubits 12 --qdepth 6
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (required)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: from config)"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=None,
        help="Override number of qubits (default: from config)"
    )
    parser.add_argument(
        "--qdepth",
        type=int,
        default=None,
        help="Override quantum circuit depth (default: from config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override device (default: from config)"
    )
    parser.add_argument(
        "--lr-g",
        type=float,
        default=None,
        help="Override generator learning rate (default: from config)"
    )
    parser.add_argument(
        "--lr-d",
        type=float,
        default=None,
        help="Override discriminator learning rate (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("🌌 QUANTUM CONDITIONAL GAN - TRAINING")
    print("=" * 70)
    print(f"📋 Loading config from: {config_path}\n")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply parameter overrides
    if args.epochs is not None:
        config_dict['num_epochs'] = args.epochs
        print(f"⚙️  Overriding epochs: {args.epochs}")
    
    if args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
        print(f"⚙️  Overriding batch size: {args.batch_size}")
    
    if args.qubits is not None:
        config_dict['num_qubits'] = args.qubits
        print(f"⚙️  Overriding qubits: {args.qubits}")
    
    if args.qdepth is not None:
        config_dict['quantum_depth'] = args.qdepth
        print(f"⚙️  Overriding quantum depth: {args.qdepth}")
    
    if args.device is not None:
        config_dict['device'] = args.device
        print(f"⚙️  Overriding device: {args.device}")
    
    if args.lr_g is not None:
        config_dict['learning_rate_g'] = args.lr_g
        print(f"⚙️  Overriding generator LR: {args.lr_g}")
    
    if args.lr_d is not None:
        config_dict['learning_rate_d'] = args.lr_d
        print(f"⚙️  Overriding discriminator LR: {args.lr_d}")
    
    # Create config object
    # Convert config dict to ensure proper types
    if isinstance(config_dict.get('eps'), str):
        config_dict['eps'] = float(config_dict['eps'])
    if isinstance(config_dict.get('learning_rate_g'), str):
        config_dict['learning_rate_g'] = float(config_dict['learning_rate_g'])
    if isinstance(config_dict.get('learning_rate_d'), str):
        config_dict['learning_rate_d'] = float(config_dict['learning_rate_d'])
    
    config = QuantumCGANConfig(**config_dict)
    
    # Print configuration summary
    print("\n" + "=" * 70)
    print("⚛️  QUANTUM CONFIGURATION")
    print("=" * 70)
    print(f"Qubits:          {config.num_qubits}")
    print(f"Circuit Depth:   {config.quantum_depth}")
    print(f"Feature Dim:     {config.quantum_feature_dim}")
    print(f"Measurement:     {config.measurement_samples} shots")
    
    print("\n" + "=" * 70)
    print("🎛️  CLASSICAL ARCHITECTURE")
    print("=" * 70)
    print(f"Latent Dim:      {config.z_dim}")
    print(f"Hidden Dim:      {config.hidden_dim}")
    print(f"Classes:         {config.num_classes}")
    print(f"Image Size:      {config.img_size}x{config.img_size}")
    
    print("\n" + "=" * 70)
    print("📚 TRAINING PARAMETERS")
    print("=" * 70)
    print(f"Epochs:          {config.num_epochs}")
    print(f"Batch Size:      {config.batch_size}")
    print(f"LR Generator:    {config.learning_rate_g}")
    print(f"LR Discriminator: {config.learning_rate_d}")
    print(f"Device:          {config.device}")
    print(f"Output:          {config.output_dir}")
    print("=" * 70 + "\n")
    
    # Start training
    try:
        train_qcgan(config)
        print("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
