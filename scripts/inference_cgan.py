#!/usr/bin/env python
"""
CGAN Inference Script - Generate synthetic defect images

Usage:
    python inference_cgan.py --mode single-class --class_id 0 --num_samples 36
    python inference_cgan.py --mode all-classes --num_samples 6
    python inference_cgan.py --mode interpolate --class_id 0 --num_samples 10
    python inference_cgan.py --mode batch --num_samples 60
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.train_cgan import Generator, Discriminator


def load_checkpoint(checkpoint_path, device):
    """Load a checkpoint and return generator."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def generate_samples_single_class(generator, class_id, num_samples, z_dim, num_classes, device):
    """Generate samples for a single class."""
    generator.eval()
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(num_samples, z_dim, device=device)
        
        # Create class labels
        c = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
        
        # Generate images
        fake_images = generator(z, c)
    
    return fake_images


def generate_samples_all_classes(generator, num_samples_per_class, z_dim, num_classes, device):
    """Generate samples for all classes."""
    all_images = []
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    for class_id in range(num_classes):
        fake_images = generate_samples_single_class(
            generator, class_id, num_samples_per_class, z_dim, num_classes, device
        )
        all_images.append(fake_images)
    
    return torch.cat(all_images, dim=0), class_names


def interpolate_latent_space(generator, class_id, num_steps, z_dim, num_classes, device):
    """Interpolate in latent space between two random vectors."""
    generator.eval()
    with torch.no_grad():
        # Two random starting points
        z_start = torch.randn(1, z_dim, device=device)
        z_end = torch.randn(1, z_dim, device=device)
        
        # Interpolate
        images = []
        for alpha in np.linspace(0, 1, num_steps):
            z_interp = (1 - alpha) * z_start + alpha * z_end
            c = torch.full((1,), class_id, dtype=torch.long, device=device)
            fake_image = generator(z_interp, c)
            images.append(fake_image)
        
        images = torch.cat(images, dim=0)
    
    return images


def save_grid_image(images, output_path, nrow=6):
    """Save images as a grid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize to [0, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    # Create grid
    grid = make_grid(images, nrow=nrow, normalize=False)
    
    # Convert to PIL and save
    grid_img = (grid[0] * 255).byte().cpu().numpy()
    Image.fromarray(grid_img, mode='L').save(output_path)
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='CGAN Inference')
    parser.add_argument('--mode', type=str, default='single-class',
                       choices=['single-class', 'all-classes', 'interpolate', 'batch'],
                       help='Generation mode')
    parser.add_argument('--class_id', type=int, default=0, help='Class ID for single-class mode')
    parser.add_argument('--num_samples', type=int, default=36, help='Number of samples to generate')
    parser.add_argument('--checkpoint', type=str, 
                       default='runs/cgan_baseline_128/checkpoints/checkpoint_epoch_0020.pt',
                       help='Path to checkpoint')
    parser.add_argument('--config', type=str, 
                       default='configs/cgan_baseline_128.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    z_dim = config.get('latent_dim', 100)
    base_channels = config.get('base_channels', 64)
    num_classes = config['num_classes']
    img_size = config['img_size']
    
    # Create generator
    generator = Generator(
        latent_dim=z_dim,
        num_classes=num_classes,
        base_channels=base_channels,
        img_size=img_size
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['generator_state'])
    generator.eval()
    
    # Create output directory
    output_dir = Path('inference_outputs/cgan')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    print(f"\n{'='*60}")
    print(f"CGAN Inference - Mode: {args.mode}")
    print(f"{'='*60}\n")
    
    if args.mode == 'single-class':
        print(f"Generating {args.num_samples} samples for class: {class_names[args.class_id]}")
        images = generate_samples_single_class(
            generator, args.class_id, args.num_samples, z_dim, num_classes, device
        )
        
        nrow = min(6, args.num_samples)
        output_path = output_dir / f'cgan_single_class_{class_names[args.class_id]}.png'
        save_grid_image(images, output_path, nrow=nrow)
    
    elif args.mode == 'all-classes':
        print(f"Generating {args.num_samples} samples per class (all 6 classes)")
        images, names = generate_samples_all_classes(
            generator, args.num_samples, z_dim, num_classes, device
        )
        
        output_path = output_dir / 'cgan_all_classes.png'
        save_grid_image(images, output_path, nrow=args.num_samples)
    
    elif args.mode == 'interpolate':
        print(f"Interpolating latent space for class: {class_names[args.class_id]}")
        images = interpolate_latent_space(
            generator, args.class_id, args.num_samples, z_dim, num_classes, device
        )
        
        output_path = output_dir / f'cgan_interpolate_{class_names[args.class_id]}.png'
        save_grid_image(images, output_path, nrow=args.num_samples)
    
    elif args.mode == 'batch':
        print(f"Generating batch: {args.num_samples} samples per class")
        for class_id in range(num_classes):
            images = generate_samples_single_class(
                generator, class_id, args.num_samples, z_dim, num_classes, device
            )
            
            class_output_dir = output_dir / 'batch' / class_names[class_id]
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual images
            for i, img in enumerate(images):
                save_path = class_output_dir / f'sample_{i:03d}.png'
                img_normalized = (img + 1) / 2
                img_normalized = img_normalized.clamp(0, 1)
                save_image(img_normalized[0], save_path)
            
            print(f"  ✅ Saved {args.num_samples} samples for class: {class_names[class_id]}")
    
    print(f"\n{'='*60}")
    print(f"✅ Inference complete!")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
