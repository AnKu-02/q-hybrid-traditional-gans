#!/usr/bin/env python3
"""
Quick test script for QStyleGAN model
Tests instantiation and forward pass
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.qstylegan import QStyleGAN, QuantumStyleProcessor


def test_quantum_processor():
    """Test quantum style processor"""
    print("🧪 Testing QuantumStyleProcessor...")
    
    processor = QuantumStyleProcessor(latent_dim=512, n_qubits=8)
    z = torch.randn(4, 512)
    
    output = processor(z)
    
    assert output.shape == z.shape, f"Expected {z.shape}, got {output.shape}"
    print(f"  ✓ Input: {z.shape} → Output: {output.shape}")
    print(f"  ✓ PASSED\n")


def test_qstylegan():
    """Test QStyleGAN model"""
    print("🧪 Testing QStyleGAN...")
    
    # Create model
    model = QStyleGAN(
        latent_dim=512,
        style_dim=512,
        n_classes=6,
        max_resolution=128,
        use_quantum=False,  # Use classical for faster test
        n_qubits=8
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"  Device: {device}")
    
    # Test generator
    print("  Testing Generator...")
    z = torch.randn(4, 512, device=device)
    c = torch.tensor([0, 1, 2, 3], device=device)
    
    images = model.generator(z, c)
    assert images.shape == (4, 3, 128, 128), f"Expected (4, 3, 128, 128), got {images.shape}"
    print(f"    ✓ Generated: {images.shape}")
    print(f"    ✓ Value range: [{images.min():.4f}, {images.max():.4f}]")
    
    # Test discriminator
    print("  Testing Discriminator...")
    scores, class_embed = model.discriminator(images, c)
    assert scores.shape == (4, 1), f"Expected (4, 1), got {scores.shape}"
    print(f"    ✓ Scores: {scores.shape}")
    print(f"    ✓ Class embed: {class_embed.shape}")
    
    # Test generation method
    print("  Testing generation method...")
    samples = model.generate(batch_size=8, device=device)
    assert samples.shape == (8, 3, 128, 128), f"Expected (8, 3, 128, 128), got {samples.shape}"
    print(f"    ✓ Samples: {samples.shape}")
    
    print(f"  ✓ PASSED\n")


def test_style_mapping():
    """Test style mapping network"""
    print("🧪 Testing Style Mapping Network...")
    
    model = QStyleGAN(
        latent_dim=512,
        style_dim=512,
        n_classes=6,
        max_resolution=128,
        use_quantum=False
    )
    
    z = torch.randn(4, 512)
    w = model.generator.style_mapping(z)
    
    assert w.shape == (4, 512), f"Expected (4, 512), got {w.shape}"
    print(f"  ✓ Latent: {z.shape} → Style: {w.shape}")
    print(f"  ✓ PASSED\n")


def test_progressive_layers():
    """Test progressive synthesis layers"""
    print("🧪 Testing Progressive Synthesis Layers...")
    
    model = QStyleGAN(
        latent_dim=512,
        style_dim=512,
        n_classes=6,
        max_resolution=128,
        use_quantum=False
    )
    
    # Check layer names
    layer_names = list(model.generator.synthesis_layers.keys())
    print(f"  Layers: {len(layer_names)}")
    for name in layer_names:
        print(f"    • {name}")
    
    print(f"  ✓ PASSED\n")


def main():
    print("=" * 60)
    print("🚀 QStyleGAN Model Tests")
    print("=" * 60 + "\n")
    
    try:
        test_quantum_processor()
        test_style_mapping()
        test_progressive_layers()
        test_qstylegan()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
