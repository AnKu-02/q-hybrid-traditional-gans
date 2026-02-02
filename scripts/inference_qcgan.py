"""
Quantum Conditional GAN - Inference & Generation
=================================================

Generate synthetic defect images using trained quantum models.

Usage:
    # Generate samples for all classes
    python scripts/inference_qcgan.py \\
        --checkpoint runs/qcgan_baseline_128/checkpoints/epoch_0020.pt \\
        --config configs/qcgan_baseline_128.yaml \\
        --mode all-classes
    
    # Generate specific class
    python scripts/inference_qcgan.py \\
        --checkpoint runs/qcgan_baseline_128/checkpoints/epoch_0020.pt \\
        --config configs/qcgan_baseline_128.yaml \\
        --class-id 0 \\
        --num-samples 100
    
    # Generate with interpolation
    python scripts/inference_qcgan.py \\
        --checkpoint runs/qcgan_baseline_128/checkpoints/epoch_0020.pt \\
        --config configs/qcgan_baseline_128.yaml \\
        --mode interpolate \\
        --num-steps 10
"""

import argparse
import torch
import yaml
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))

from train_qcgan import QuantumCGANConfig, QuantumGenerator


class QuantumCGANInference:
    """Inference handler for Quantum CGAN"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = QuantumCGANConfig(**config_dict)
        
        # Load model
        self.generator = QuantumGenerator(
            z_dim=self.config.z_dim,
            num_qubits=self.config.num_qubits,
            hidden_dim=self.config.hidden_dim,
            quantum_depth=self.config.quantum_depth,
            num_classes=self.config.num_classes
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.generator.eval()
        
        print(f"✅ Loaded quantum generator from: {checkpoint_path}")
        print(f"   Epoch: {checkpoint['epoch']}")
    
    def generate(self, class_id: int, num_samples: int = 36) -> torch.Tensor:
        """Generate samples for specific class"""
        print(f"\n🌌 Generating {num_samples} samples for class {class_id}...")
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.z_dim, device=self.device)
            class_ids = torch.full((num_samples,), class_id, dtype=torch.long, device=self.device)
            samples = self.generator(z, class_ids)
        
        return samples
    
    def generate_all_classes(self, samples_per_class: int = 36) -> torch.Tensor:
        """Generate samples for all classes"""
        print(f"\n🌌 Generating {samples_per_class} samples per class...")
        
        all_samples = []
        for class_id in tqdm(range(self.config.num_classes), desc="Generating classes"):
            samples = self.generate(class_id, samples_per_class)
            all_samples.append(samples)
        
        return torch.cat(all_samples, dim=0)
    
    def interpolate(self, class_id: int, num_steps: int = 10) -> torch.Tensor:
        """Interpolate in latent space"""
        print(f"\n🌌 Interpolating latent space for class {class_id} ({num_steps} steps)...")
        
        with torch.no_grad():
            # Two random points in latent space
            z1 = torch.randn(1, self.config.z_dim, device=self.device)
            z2 = torch.randn(1, self.config.z_dim, device=self.device)
            
            class_ids = torch.full((1,), class_id, dtype=torch.long, device=self.device)
            
            samples = []
            for t in tqdm(np.linspace(0, 1, num_steps), desc="Interpolating"):
                z_interp = (1 - t) * z1 + t * z2
                sample = self.generator(z_interp, class_ids)
                samples.append(sample)
        
        return torch.cat(samples, dim=0)
    
    def quantum_effect_test(self, class_id: int, num_samples: int = 9) -> torch.Tensor:
        """Compare quantum vs classical-only (ablation study)"""
        print(f"\n⚛️  Testing quantum effect for class {class_id}...")
        
        # Generate with quantum
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.z_dim, device=self.device)
            class_ids = torch.full((num_samples,), class_id, dtype=torch.long, device=self.device)
            quantum_samples = self.generator(z, class_ids)
        
        return quantum_samples
    
    def batch_generate(self, num_samples: int = 1000, save_dir: str = None) -> None:
        """Generate large batch and save to disk"""
        print(f"\n🌌 Batch generating {num_samples} quantum images...")
        
        if save_dir is None:
            save_dir = Path(f"runs/qcgan_generated_{num_samples}")
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class subdirectories
        class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
        for class_name in class_names:
            (save_dir / class_name).mkdir(exist_ok=True)
        
        # Generate
        samples_per_class = num_samples // self.config.num_classes
        
        for class_id in tqdm(range(self.config.num_classes), desc="Classes"):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            with torch.no_grad():
                for i in tqdm(range(0, samples_per_class, 32), desc=f"  {class_name}", leave=False):
                    batch_size = min(32, samples_per_class - i)
                    z = torch.randn(batch_size, self.config.z_dim, device=self.device)
                    class_ids = torch.full((batch_size,), class_id, dtype=torch.long, device=self.device)
                    samples = self.generator(z, class_ids)
                    
                    # Save images
                    for j, sample in enumerate(samples):
                        img = sample[0].cpu().numpy()  # (128, 128)
                        img = ((img + 1) / 2 * 255).astype(np.uint8)  # Normalize to [0, 255]
                        
                        pil_img = Image.fromarray(img, mode='L')
                        filename = f"{class_name}_{i + j:05d}.png"
                        pil_img.save(save_dir / class_name / filename)
        
        print(f"\n✅ Generated {num_samples} quantum images saved to: {save_dir}")


def save_image_grid(images: torch.Tensor, path: Path, num_classes: int = 6) -> None:
    """Save grid of images"""
    from torchvision.utils import make_grid
    
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    grid = make_grid(images, nrow=num_classes, normalize=False)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    # Handle both grayscale and RGB grids
    if grid.shape[0] == 1:
        # Single channel - already grayscale
        grid_img = (grid[0] * 255).byte().cpu().numpy()
    else:
        # Multi-channel (RGB) - take first channel only
        grid_img = (grid[0] * 255).byte().cpu().numpy()
    
    Image.fromarray(grid_img, mode='L').save(path)
    print(f"✅ Grid saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Quantum CGAN Inference")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single-class",
        choices=["single-class", "all-classes", "interpolate", "quantum-effect", "batch"],
        help="Generation mode"
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class ID for single-class mode (0-5)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=36,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of interpolation steps"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("⚛️  QUANTUM CONDITIONAL GAN - INFERENCE")
    print("=" * 70)
    
    # Initialize inference
    inference = QuantumCGANInference(
        args.checkpoint,
        args.config,
        device=args.device
    )
    
    # Generate based on mode
    if args.mode == "single-class":
        samples = inference.generate(args.class_id, args.num_samples)
        
        output_path = Path("quantum_samples_single.png")
        save_image_grid(samples[:36], output_path, inference.config.num_classes)
    
    elif args.mode == "all-classes":
        samples = inference.generate_all_classes(args.num_samples // inference.config.num_classes)
        
        output_path = Path("quantum_samples_all_classes.png")
        save_image_grid(samples[:36], output_path, inference.config.num_classes)
    
    elif args.mode == "interpolate":
        samples = inference.interpolate(args.class_id, args.num_steps)
        
        output_path = Path("quantum_samples_interpolation.png")
        save_image_grid(samples, output_path, min(6, args.num_steps))
    
    elif args.mode == "quantum-effect":
        samples = inference.quantum_effect_test(args.class_id, 9)
        
        output_path = Path("quantum_samples_effect_test.png")
        save_image_grid(samples, output_path, 3)
    
    elif args.mode == "batch":
        inference.batch_generate(args.num_samples, args.save_dir)
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
