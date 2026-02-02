#!/usr/bin/env python3
"""
Inference script for QStyleGAN

Generate defect images using trained QStyleGAN model.

Usage:
    python scripts/inference_qstylegan.py \\
        --checkpoint runs/qstylegan_baseline_128/checkpoints/best.pt \\
        --output results/qstylegan_samples \\
        --num-samples 100
"""

import argparse
import torch
import torchvision.utils as vutils
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.qstylegan import QStyleGAN


class QStyleGANInference:
    """QStyleGAN inference engine"""
    
    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device = torch.device('cpu')
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        print(f"📥 Loading model from {checkpoint_path}")
        self.model = QStyleGAN.load(self.checkpoint_path, device=device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Config: latent_dim={self.model.latent_dim}, "
              f"n_classes={self.model.n_classes}, "
              f"max_res={self.model.max_resolution}")
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 32,
        per_class: bool = True,
        truncation: float = 1.0,
        seed: int = None
    ) -> torch.Tensor:
        """
        Generate images
        
        Args:
            num_samples: Number of samples to generate
            per_class: If True, generate equal samples per class
            truncation: Truncation trick value (0.0-1.0)
            seed: Random seed for reproducibility
        
        Returns:
            Generated images tensor (N, 3, H, W)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        z = torch.randn(
            num_samples,
            self.model.latent_dim,
            device=self.device
        )
        
        if per_class:
            # Generate equal samples per class
            num_per_class = num_samples // self.model.n_classes
            classes = torch.tensor(
                [i for i in range(self.model.n_classes) 
                 for _ in range(num_per_class)],
                device=self.device
            )
            # Adjust for remainder
            if num_samples % self.model.n_classes != 0:
                remainder = num_samples % self.model.n_classes
                extra_classes = torch.randint(0, self.model.n_classes, (remainder,), device=self.device)
                classes = torch.cat([classes, extra_classes])
        else:
            # Random classes
            classes = torch.randint(0, self.model.n_classes, (num_samples,), device=self.device)
        
        # Generate images
        images = self.model.generator(z, classes, truncation=truncation)
        
        return images
    
    def save_grid(
        self,
        images: torch.Tensor,
        output_path: Path,
        nrow: int = 8,
        normalize: bool = True
    ):
        """Save images as grid"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        grid = vutils.make_grid(
            images,
            nrow=nrow,
            normalize=normalize,
            value_range=(-1, 1)
        )
        
        # Save grid
        vutils.save_image(grid, output_path)
        print(f"✓ Saved grid: {output_path}")
    
    def save_individual(
        self,
        images: torch.Tensor,
        output_dir: Path,
        class_names: list = None
    ):
        """Save individual images"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if class_names is None:
            class_names = [f'class_{i}' for i in range(self.model.n_classes)]
        
        for i, img in enumerate(images):
            # Denormalize
            img_denorm = (img + 1) / 2  # [-1, 1] -> [0, 1]
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            # Convert to PIL
            img_pil = vutils.transforms.ToPILImage()(img_denorm.cpu())
            
            # Save
            img_path = output_dir / f'sample_{i:04d}.png'
            img_pil.save(img_path)
        
        print(f"✓ Saved {len(images)} individual images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using QStyleGAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 samples in batches by class
  python scripts/inference_qstylegan.py \\
    --checkpoint runs/qstylegan_baseline_128/checkpoints/best.pt \\
    --num-samples 100 \\
    --output results/qstylegan_samples

  # Generate with truncation trick for diversity control
  python scripts/inference_qstylegan.py \\
    --checkpoint runs/qstylegan_baseline_128/checkpoints/best.pt \\
    --num-samples 50 \\
    --truncation 0.7 \\
    --output results/qstylegan_truncated
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/qstylegan_samples',
        help='Output directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--truncation',
        type=float,
        default=1.0,
        help='Truncation value (0.0-1.0, lower = more consistent)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--per-class',
        action='store_true',
        default=True,
        help='Generate equal samples per class'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Inference engine
    engine = QStyleGANInference(
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # Generate samples
    print(f"\n🎨 Generating {args.num_samples} samples...")
    output_dir = Path(args.output)
    
    # Generate in batches
    all_images = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)
        
        images = engine.generate(
            num_samples=batch_size,
            per_class=args.per_class,
            truncation=args.truncation,
            seed=args.seed + batch_idx if args.seed else None
        )
        
        all_images.append(images.cpu())
        print(f"  ✓ Generated batch {batch_idx + 1}/{num_batches}")
    
    # Combine all batches
    all_images = torch.cat(all_images, dim=0)
    
    # Save grid
    grid_path = output_dir / 'samples_grid.png'
    engine.save_grid(all_images, grid_path, nrow=10)
    
    # Save individual images
    individual_dir = output_dir / 'samples_individual'
    engine.save_individual(all_images, individual_dir)
    
    # Save summary
    summary_path = output_dir / 'generation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"QStyleGAN Generation Summary\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Total Samples: {len(all_images)}\n")
        f.write(f"Truncation: {args.truncation}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nOutput Files:\n")
        f.write(f"  • Grid: {grid_path}\n")
        f.write(f"  • Individual: {individual_dir}\n")
    
    print(f"\n✅ Generation complete!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == '__main__':
    main()
