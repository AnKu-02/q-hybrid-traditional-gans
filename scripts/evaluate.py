#!/usr/bin/env python
"""
Main evaluation script for GANs.

Comprehensive evaluation comparing real and generated images using:
1. FID - measures distribution similarity
2. Label Fidelity - measures class conditioning adherence
3. Classifier Performance - baseline on real data

Usage:
    python scripts/evaluate.py --model cgan --run_name cgan_baseline_128 --config configs/cgan_baseline_128.yaml
    python scripts/evaluate.py --model qcgan --run_name qcgan_baseline_128 --config configs/qcgan_baseline_128.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.evaluate import GANEvaluator, EvalConfig
from train.train_cgan import Generator as CGANGenerator


def load_cgan_model(checkpoint_path: str, config: dict, device: str):
    """Load CGAN generator from checkpoint"""
    from train.train_cgan import Generator
    
    z_dim = config.get('latent_dim', 100)
    num_classes = config.get('num_classes', 6)
    base_channels = config.get('base_channels', 64)
    img_size = config.get('img_size', 128)
    
    generator = Generator(
        latent_dim=z_dim,
        num_classes=num_classes,
        base_channels=base_channels,
        img_size=img_size
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state'])
    generator.eval()
    
    return generator, z_dim


def load_qcgan_model(checkpoint_path: str, config: dict, device: str):
    """Load QCGAN generator from checkpoint"""
    from train.train_qcgan import QuantumGenerator
    
    z_dim = config.get('z_dim', 32)
    hidden_dim = config.get('hidden_dim', 256)
    num_classes = config.get('num_classes', 6)
    num_qubits = config.get('num_qubits', 8)
    quantum_depth = config.get('quantum_depth', 4)
    
    generator = QuantumGenerator(
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_qubits=num_qubits,
        quantum_depth=quantum_depth
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    return generator, z_dim


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GAN model quality using FID and label fidelity"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["cgan", "qcgan"],
        required=True,
        help="Model type to evaluate"
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name (e.g., 'cgan_baseline_128', 'qcgan_baseline_128')"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (if None, uses default from run_name)"
    )
    
    parser.add_argument(
        "--num_images_per_class",
        type=int,
        default=50,
        help="Number of images to generate per class"
    )
    
    parser.add_argument(
        "--classifier_epochs",
        type=int,
        default=10,
        help="Epochs to train classifier"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    print("\n" + "="*70)
    print(f"🔬 GAN EVALUATION: {args.model.upper()}")
    print("="*70)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print(f"\n✓ Loaded config: {args.config}")
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = "cpu"
    
    print(f"✓ Using device: {device}")
    
    # Load checkpoint
    if args.checkpoint is None:
        run_dir = Path("runs") / args.run_name
        checkpoint_path = run_dir / "checkpoints" / "epoch_0020.pt"
        
        if not checkpoint_path.exists():
            # Try alternative checkpoint names
            checkpoints = sorted(run_dir.glob("checkpoints/*.pt"))
            if checkpoints:
                checkpoint_path = checkpoints[-1]  # Use latest
            else:
                raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    else:
        checkpoint_path = args.checkpoint
    
    print(f"✓ Using checkpoint: {checkpoint_path}")
    
    # Load model
    print(f"\n✓ Loading {args.model.upper()} generator...")
    if args.model == "cgan":
        generator, z_dim = load_cgan_model(
            str(checkpoint_path),
            config,
            device
        )
    else:  # qcgan
        generator, z_dim = load_qcgan_model(
            str(checkpoint_path),
            config,
            device
        )
    
    print(f"✓ Generator loaded (z_dim={z_dim})")
    
    # Get class names
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    num_classes = config.get('num_classes', 6)
    
    # Create evaluator config
    image_dir = config.get('image_dir', 'data/NEU_baseline_128')
    
    eval_config = EvalConfig(
        train_image_dir=f"{image_dir}/train",
        val_image_dir=f"{image_dir}/validation",
        num_images_per_class=args.num_images_per_class,
        classifier_epochs=args.classifier_epochs,
        classifier_batch_size=args.batch_size,
        fid_batch_size=args.batch_size,
        device=device,
        output_dir=f"runs/{args.run_name}/evaluation"
    )
    
    print(f"✓ Evaluation config ready")
    print(f"  - Train images: {eval_config.train_image_dir}")
    print(f"  - Val images: {eval_config.val_image_dir}")
    print(f"  - Output: {eval_config.output_dir}")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    
    evaluator = GANEvaluator(eval_config, class_names, num_classes)
    
    metrics = evaluator.evaluate(generator, z_dim, args.run_name)
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("📈 EVALUATION RESULTS")
    print("="*70)
    
    print(f"\n📐 FID Score: {metrics.overall_fid:.4f}")
    print("   (Lower is better - measures similarity of feature distributions)")
    
    print(f"\n✅ Label Fidelity: {metrics.label_fidelity:.4f}")
    print("   (% of generated images classified as their conditioning label)")
    
    print(f"\n🎯 Classifier Accuracy (on real validation): {metrics.classifier_real_accuracy:.4f}")
    print("   (Baseline classifier performance - higher = more reliable fidelity)")
    
    print(f"\n📊 Per-Class Label Fidelity:")
    for class_name, fidelity in metrics.per_class_label_fidelity.items():
        print(f"   {class_name:20s}: {fidelity:.4f}")
    
    print(f"\n🏆 Per-Class Precision (Real Validation):")
    for class_name, precision in metrics.classifier_real_precision.items():
        print(f"   {class_name:20s}: {precision:.4f}")
    
    print(f"\n🎪 Per-Class Recall (Real Validation):")
    for class_name, recall in metrics.classifier_real_recall.items():
        print(f"   {class_name:20s}: {recall:.4f}")
    
    print(f"\n📦 Dataset Info:")
    print(f"   Generated images: {metrics.num_generated_images} ({args.num_images_per_class} per class)")
    print(f"   Real validation images: {metrics.num_real_validation_images}")
    
    # Save metrics
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)
    
    evaluator.save_metrics(metrics, args.run_name)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {eval_config.output_dir}/")
    print(f"  - metrics.json (detailed metrics)")
    print(f"  - metrics.csv (tabular format)")
    print(f"  - generated/ (generated images by class)")


if __name__ == "__main__":
    main()
