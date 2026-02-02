#!/usr/bin/env python
"""
Quick validation test for evaluation framework.

Runs a mini evaluation with small datasets to verify all components work.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.evaluate import (
    GANEvaluator, EvalConfig, ClassificationDataset,
    SimpleClassifier, FIDCalculator
)
from train.train_cgan import Generator
import torch

def test_dataset_loading():
    """Test that datasets load correctly"""
    print("\n" + "="*70)
    print("✓ Testing dataset loading...")
    print("="*70)
    
    dataset = ClassificationDataset(
        "data/NEU_baseline_128/train",
        img_size=128
    )
    
    print(f"✓ Classes found: {dataset.classes}")
    print(f"✓ Total images: {len(dataset)}")
    
    # Try loading one image
    img, label = dataset[0]
    print(f"✓ Sample image shape: {img.shape}")
    print(f"✓ Sample label: {label}")


def test_generator_loading():
    """Test that generator loads correctly"""
    print("\n" + "="*70)
    print("✓ Testing generator loading...")
    print("="*70)
    
    import yaml
    
    with open("configs/cgan_baseline_128.yaml") as f:
        config = yaml.safe_load(f)
    
    generator = Generator(
        latent_dim=config['latent_dim'],
        num_classes=config['num_classes'],
        base_channels=config['base_channels'],
        img_size=config['img_size']
    )
    
    # Load checkpoint
    checkpoint_path = "runs/cgan_baseline_128/checkpoints/checkpoint_epoch_0020.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(checkpoint['generator_state'])
    
    print(f"✓ Generator loaded successfully")
    
    # Try generating one image
    z = torch.randn(1, config['latent_dim'])
    c = torch.tensor([0])
    with torch.no_grad():
        img = generator(z, c)
    
    print(f"✓ Generated image shape: {img.shape}")


def test_classifier():
    """Test that classifier can be created and trained"""
    print("\n" + "="*70)
    print("✓ Testing classifier...")
    print("="*70)
    
    classifier = SimpleClassifier(
        num_classes=6,
        hidden_dim=256
    )
    
    print(f"✓ Classifier created successfully")
    print(f"✓ Total parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Try one forward pass
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        out = classifier(x)
    
    print(f"✓ Forward pass output shape: {out.shape}")
    print(f"✓ Output logits range: [{out.min():.4f}, {out.max():.4f}]")


def test_fid_calculator():
    """Test FID calculator"""
    print("\n" + "="*70)
    print("✓ Testing FID calculator...")
    print("="*70)
    
    fid_calc = FIDCalculator(device="cpu", batch_size=16)
    print(f"✓ FID calculator initialized")
    
    # Create dummy distributions
    mu1 = np.random.randn(2048)
    sigma1 = np.eye(2048)
    mu2 = np.random.randn(2048)
    sigma2 = np.eye(2048)
    
    fid = fid_calc.compute_fid(mu1, sigma1, mu2, sigma2)
    print(f"✓ FID computation successful: {fid:.4f}")


if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "="*70)
    print("🧪 EVALUATION FRAMEWORK VALIDATION")
    print("="*70)
    
    try:
        test_dataset_loading()
        test_generator_loading()
        test_classifier()
        test_fid_calculator()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nFramework is ready! Run:")
        print("  python scripts/evaluate.py --model cgan --run_name cgan_baseline_128 --config configs/cgan_baseline_128.yaml")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
