#!/usr/bin/env python3
"""
StyleGAN2-ADA Inference Script
==============================

Generate synthetic images and latent space interpolations using trained StyleGAN2-ADA model.

Usage:
    # Generate images for specific class
    python scripts/inference_stylegan2_ada.py \\
        --checkpoint runs/stylegan2_ada_baseline_128/checkpoints/epoch_0020.pt \\
        --class-id 0 \\
        --num-samples 36 \\
        --output generated_samples.png
    
    # Style mixing interpolation
    python scripts/inference_stylegan2_ada.py \\
        --checkpoint runs/stylegan2_ada_baseline_128/checkpoints/epoch_0020.pt \\
        --style-mixing \\
        --output style_mixing.png
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train.train_stylegan2_ada import StyleGAN2Generator, StyleGAN2ADAConfig
import yaml


class StyleGAN2ADAInference:
    """Inference helper for StyleGAN2-ADA"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.config = StyleGAN2ADAConfig(**config_dict)
        else:
            # Try to find config in checkpoint directory
            checkpoint_dir = Path(checkpoint_path).parent.parent
            config_path = checkpoint_dir / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                self.config = StyleGAN2ADAConfig(**config_dict)
            else:
                raise ValueError("Config file not found. Provide with --config")
        
        # Load generator
        self.generator = StyleGAN2Generator(
            z_dim=self.config.z_dim,
            w_dim=self.config.w_dim,
            img_size=self.config.img_size,
            num_classes=self.config.num_classes
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator.eval()
        
        print(f"✅ Loaded model from: {checkpoint_path}")
    
    def generate(self, class_id: int, num_samples: int = 36) -> torch.Tensor:
        """Generate images for specific class"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.z_dim, device=self.device)
            class_ids = torch.full((num_samples,), class_id, device=self.device, dtype=torch.long)
            images = self.generator(z, class_ids)
        
        return images
    
    def generate_all_classes(self, num_per_class: int = 6) -> torch.Tensor:
        """Generate images for all classes"""
        all_images = []
        
        with torch.no_grad():
            for class_id in range(self.config.num_classes):
                z = torch.randn(num_per_class, self.config.z_dim, device=self.device)
                class_ids = torch.full((num_per_class,), class_id, device=self.device, dtype=torch.long)
                images = self.generator(z, class_ids)
                all_images.append(images)
        
        return torch.cat(all_images, dim=0)
    
    def style_mixing(self, num_samples: int = 6) -> torch.Tensor:
        """Generate style mixing samples"""
        images = []
        
        with torch.no_grad():
            for class_id in range(self.config.num_classes):
                # Source latent codes
                z1 = torch.randn(num_samples, self.config.z_dim, device=self.device)
                z2 = torch.randn(num_samples, self.config.z_dim, device=self.device)
                
                class_ids = torch.full((num_samples,), class_id, device=self.device, dtype=torch.long)
                
                # Generate images
                img1 = self.generator(z1, class_ids)
                img2 = self.generator(z2, class_ids)
                
                # Interpolate (simple linear mixing)
                for alpha in [0.0, 0.5, 1.0]:
                    z_mixed = z1 * alpha + z2 * (1 - alpha)
                    img_mixed = self.generator(z_mixed, class_ids)
                    images.append(img_mixed)
        
        return torch.cat(images, dim=0)
    
    def interpolate(self, class_id: int, num_steps: int = 10) -> torch.Tensor:
        """Linear interpolation in latent space"""
        images = []
        
        with torch.no_grad():
            z1 = torch.randn(1, self.config.z_dim, device=self.device)
            z2 = torch.randn(1, self.config.z_dim, device=self.device)
            
            class_ids = torch.full((1,), class_id, device=self.device, dtype=torch.long)
            
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z = z1 * alpha + z2 * (1 - alpha)
                img = self.generator(z, class_ids)
                images.append(img)
        
        return torch.cat(images, dim=0)


def save_image_grid(images: torch.Tensor, path: str, nrow: int = None) -> None:
    """Save image grid to file"""
    # Denormalize
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    if nrow is None:
        nrow = int(np.sqrt(images.size(0)))
    
    # Create grid
    grid = make_grid(images, nrow=nrow, normalize=False)
    
    # Save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    grid_img = (grid * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(grid_img.squeeze(), mode='L').save(path)
    print(f"✅ Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using trained StyleGAN2-ADA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 36 images for class 0 (crazing)
    python scripts/inference_stylegan2_ada.py \\
        --checkpoint runs/stylegan2_ada_baseline_128/checkpoints/epoch_0020.pt \\
        --class-id 0 \\
        --num-samples 36 \\
        --output generated_crazing.png
    
    # Generate images for all classes
    python scripts/inference_stylegan2_ada.py \\
        --checkpoint runs/stylegan2_ada_baseline_128/checkpoints/epoch_0020.pt \\
        --all-classes \\
        --num-per-class 6 \\
        --output all_classes.png
    
    # Latent space interpolation
    python scripts/inference_stylegan2_ada.py \\
        --checkpoint runs/stylegan2_ada_baseline_128/checkpoints/epoch_0020.pt \\
        --interpolate \\
        --class-id 0 \\
        --num-steps 10 \\
        --output interpolation.png
        """
    )
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"])
    
    # Generation modes
    parser.add_argument("--class-id", type=int, default=0, help="Class ID (0-5)")
    parser.add_argument("--num-samples", type=int, default=36, help="Number of samples to generate")
    parser.add_argument("--all-classes", action="store_true", help="Generate for all classes")
    parser.add_argument("--num-per-class", type=int, default=6, help="Samples per class")
    parser.add_argument("--interpolate", action="store_true", help="Linear interpolation in latent space")
    parser.add_argument("--num-steps", type=int, default=10, help="Interpolation steps")
    parser.add_argument("--style-mixing", action="store_true", help="Style mixing visualization")
    
    parser.add_argument("--output", type=str, default="generated.png", help="Output image path")
    
    args = parser.parse_args()
    
    # Load model
    print("📋 Loading StyleGAN2-ADA model...")
    inference = StyleGAN2ADAInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Generate
    print("🎨 Generating images...")
    
    if args.all_classes:
        print(f"  - Generating {args.num_per_class} samples per class...")
        images = inference.generate_all_classes(num_per_class=args.num_per_class)
        nrow = inference.config.num_classes
    elif args.interpolate:
        print(f"  - Linear interpolation: {args.num_steps} steps, class {args.class_id}")
        images = inference.interpolate(class_id=args.class_id, num_steps=args.num_steps)
        nrow = args.num_steps
    elif args.style_mixing:
        print(f"  - Style mixing visualization...")
        images = inference.style_mixing(num_samples=6)
        nrow = 6
    else:
        print(f"  - Generating {args.num_samples} samples for class {args.class_id}")
        images = inference.generate(class_id=args.class_id, num_samples=args.num_samples)
        nrow = 6
    
    # Save
    print(f"💾 Saving images...")
    save_image_grid(images, args.output, nrow=nrow)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
