"""
Conditional GAN (CGAN) Training Implementation

Official Paper & References:
- "Conditional Generative Adversarial Nets" (Mirza & Osindski, 2014)
  https://arxiv.org/abs/1411.1784
- Original Implementation: https://github.com/tensorlayer/dcgan

Architecture: DCGAN-style Conditional GAN
- Generator: Takes noise + one-hot class label as input
- Discriminator: Classifies real/fake + verifies class consistency
- Training: Adversarial + classification losses
"""

import os
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Optional
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# CONFIG & DATACLASSES
# ============================================================================

@dataclass
class CGANConfig:
    """Conditional GAN Configuration"""
    # Dataset
    metadata_path: str
    image_dir: str
    num_classes: int = 6
    img_size: int = 128
    
    # Training
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0002
    beta1: float = 0.5  # Adam optimizer
    beta2: float = 0.999
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model architecture
    latent_dim: int = 100
    base_channels: int = 64  # Base number of channels in generator/discriminator
    
    # Checkpointing & Logging
    sample_interval: int = 5  # Sample every N epochs
    checkpoint_interval: int = 10  # Save checkpoint every N epochs
    num_sample_images: int = 36  # Grid of 6x6 samples
    
    # Regularization
    use_gradient_penalty: bool = False
    lambda_gp: float = 10.0
    
    # Output directories (set by script)
    run_name: str = "cgan_baseline_128"
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"runs/{self.run_name}"


# ============================================================================
# DATASET
# ============================================================================

class NEUDefectDataset(Dataset):
    """NEU-DET Defect Dataset Loader"""
    
    def __init__(
        self,
        metadata_path: str,
        image_dir: str,
        img_size: int = 128,
        split: str = "train",
        classes: list = None
    ):
        """
        Args:
            metadata_path: Path to metadata.csv
            image_dir: Base directory containing images
            img_size: Target image size
            split: "train" or "validation"
            classes: List of class names
        """
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.split = split
        self.classes = classes or [
            "crazing", "inclusion", "patches", 
            "pitted_surface", "rolled-in_scale", "scratches"
        ]
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        
        # Load metadata
        self.metadata = []
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.metadata.append(row)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Range [-1, 1]
        ])
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.metadata[idx]
        image_path = self.image_dir / row['filepath']
        
        # Load and transform image
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        
        # Get class label (use class_id or parse from class_name)
        class_id = int(row['class_id']) if 'class_id' in row else self.class_to_id[row['class_name']]
        
        return image, class_id


# ============================================================================
# MODELS: Generator & Discriminator
# ============================================================================

class Generator(nn.Module):
    """
    Conditional Generator
    Input: Noise (latent_dim,) + One-hot class label (num_classes,)
    Output: Generated image (1, img_size, img_size)
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 6,
        base_channels: int = 64,
        img_size: int = 128
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.base_channels = base_channels
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Calculate initial size: img_size / 2^4 (4 deconv layers)
        self.init_size = img_size // 16  # 8 for 128x128
        
        # Dense layer: (latent_dim + latent_dim) -> (base_channels * 8 * init_size^2)
        self.dense = nn.Linear(
            latent_dim * 2,
            base_channels * 8 * self.init_size * self.init_size
        )
        
        # Deconvolutional layers (upsampling)
        self.deconv_layers = nn.Sequential(
            # (512, 8, 8) -> (256, 16, 16)
            nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            
            # (256, 16, 16) -> (128, 32, 32)
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            
            # (128, 32, 32) -> (64, 64, 64)
            nn.ConvTranspose2d(
                base_channels * 2, base_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            # (64, 64, 64) -> (1, 128, 128)
            nn.ConvTranspose2d(
                base_channels, 1,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: (batch_size, latent_dim)
            labels: (batch_size,) - class indices
        
        Returns:
            Generated images: (batch_size, 1, img_size, img_size)
        """
        # Embed labels
        label_emb = self.label_emb(labels)  # (batch, latent_dim)
        
        # Concatenate noise + label embedding
        combined = torch.cat([noise, label_emb], dim=1)  # (batch, 2*latent_dim)
        
        # Dense layer
        x = self.dense(combined)
        x = x.view(
            -1, 
            self.base_channels * 8, 
            self.init_size, 
            self.init_size
        )
        
        # Deconvolutional layers
        return self.deconv_layers(x)


class Discriminator(nn.Module):
    """
    Conditional Discriminator
    Input: Image (1, img_size, img_size) + Class label (num_classes,)
    Output: Real/Fake probability (1,)
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        base_channels: int = 64,
        img_size: int = 128
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        
        # Convolutional layers (downsampling)
        self.conv_layers = nn.Sequential(
            # (2, 128, 128) -> (64, 64, 64)
            nn.Conv2d(2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            
            # (64, 64, 64) -> (128, 32, 32)
            nn.Conv2d(
                base_channels, base_channels * 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            # (128, 32, 32) -> (256, 16, 16)
            nn.Conv2d(
                base_channels * 2, base_channels * 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            
            # (256, 16, 16) -> (512, 8, 8)
            nn.Conv2d(
                base_channels * 4, base_channels * 8,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
        )
        
        # Final layer
        self.dense = nn.Sequential(
            nn.Linear(base_channels * 8 * 8 * 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 1, img_size, img_size)
            labels: (batch_size,) - class indices
        
        Returns:
            Real/Fake probability: (batch_size, 1)
        """
        # Embed labels
        label_emb = self.label_emb(labels)  # (batch, img_size*img_size)
        label_emb = label_emb.view(-1, 1, 128, 128)  # (batch, 1, 128, 128)
        
        # Concatenate image + label
        combined = torch.cat([images, label_emb], dim=1)  # (batch, 2, 128, 128)
        
        # Convolutional layers
        x = self.conv_layers(combined)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dense layer
        return self.dense(x)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_sample_grid(
    generator: Generator,
    device: torch.device,
    num_classes: int = 6,
    latent_dim: int = 100,
    num_samples_per_class: int = 6
) -> np.ndarray:
    """
    Create a grid of generated images (one row per class).
    
    Args:
        generator: Trained generator
        device: torch device
        num_classes: Number of classes
        latent_dim: Latent dimension size
        num_samples_per_class: Samples per class (creates square grid)
    
    Returns:
        Grid image as numpy array
    """
    generator.eval()
    with torch.no_grad():
        images = []
        for class_id in range(num_classes):
            # Generate samples for this class
            noise = torch.randn(
                num_samples_per_class, latent_dim,
                device=device
            )
            labels = torch.full(
                (num_samples_per_class,),
                class_id,
                dtype=torch.long,
                device=device
            )
            
            generated = generator(noise, labels)
            images.append(generated.cpu())
        
        # Stack all images
        all_images = torch.cat(images, dim=0)
        
        # Denormalize from [-1, 1] to [0, 1]
        all_images = (all_images + 1) / 2
        all_images = torch.clamp(all_images, 0, 1)
        
        # Create grid
        grid = make_image_grid(all_images, num_samples_per_class)
        
        return grid


def make_image_grid(images: torch.Tensor, nrow: int) -> np.ndarray:
    """Create a grid of images."""
    # images: (N, 1, H, W)
    N = images.size(0)
    H, W = images.size(2), images.size(3)
    nrow = min(nrow, N)
    ncol = (N + nrow - 1) // nrow
    
    # Pad to fill grid
    pad_size = nrow * ncol - N
    if pad_size > 0:
        images = torch.cat([images, torch.zeros_like(images[:pad_size])], dim=0)
    
    # Reshape and transpose
    images = images.view(ncol, nrow, 1, H, W)
    images = images.permute(0, 3, 1, 4, 2).contiguous()
    images = images.view(ncol * H, nrow * W, 1)
    
    return images.squeeze(-1).numpy()


def save_checkpoint(
    epoch: int,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    checkpoint_dir: Path
):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save({
        'epoch': epoch,
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'optimizer_g_state': optimizer_g.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_config(config_path: str) -> CGANConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    return CGANConfig(**cfg_dict)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_cgan(config: CGANConfig):
    """
    Main training loop for Conditional GAN.
    
    Args:
        config: CGANConfig object
    """
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Device
    device = torch.device(config.device)
    print(f"Device: {device}")
    
    # Create output directories
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    log_dir = output_dir / "logs"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
    
    print(f"\n{'='*70}")
    print(f"Conditional GAN Training")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"{'='*70}\n")
    
    # ========== DATASET ==========
    train_dataset = NEUDefectDataset(
        metadata_path=config.metadata_path,
        image_dir=Path(config.metadata_path).parent,
        img_size=config.img_size,
        split="train",
        classes=["crazing", "inclusion", "patches", 
                 "pitted_surface", "rolled-in_scale", "scratches"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    print(f"Dataset loaded: {len(train_dataset)} training images")
    print(f"Batches per epoch: {len(train_loader)}\n")
    
    # ========== MODELS ==========
    generator = Generator(
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        base_channels=config.base_channels,
        img_size=config.img_size
    ).to(device)
    
    discriminator = Discriminator(
        num_classes=config.num_classes,
        base_channels=config.base_channels,
        img_size=config.img_size
    ).to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}\n")
    
    # ========== OPTIMIZERS ==========
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=config.learning_rate_g,
        betas=(config.beta1, config.beta2)
    )
    
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate_d,
        betas=(config.beta1, config.beta2)
    )
    
    # ========== LOSS FUNCTION ==========
    criterion = nn.BCELoss()
    
    # ========== LOGGING ==========
    log_file = log_dir / "train_log.csv"
    log_file.write_text("epoch,d_loss,g_loss\n")
    
    # ========== FIXED NOISE FOR SAMPLING ==========
    fixed_noise = torch.randn(
        config.num_sample_images, config.latent_dim,
        device=device
    )
    fixed_labels = torch.tensor(
        [i % config.num_classes for i in range(config.num_sample_images)],
        dtype=torch.long,
        device=device
    )
    
    # ========== TRAINING LOOP ==========
    for epoch in range(config.num_epochs):
        generator.train()
        discriminator.train()
        
        d_losses = []
        g_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for real_images, labels in pbar:
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)
            
            # ===== DISCRIMINATOR UPDATE =====
            optimizer_d.zero_grad()
            
            # Real images
            real_output = discriminator(real_images, labels)
            real_labels_d = torch.ones(batch_size, 1, device=device)
            loss_d_real = criterion(real_output, real_labels_d)
            
            # Fake images
            noise = torch.randn(batch_size, config.latent_dim, device=device)
            fake_images = generator(noise, labels)
            fake_output = discriminator(fake_images.detach(), labels)
            fake_labels_d = torch.zeros(batch_size, 1, device=device)
            loss_d_fake = criterion(fake_output, fake_labels_d)
            
            # Total discriminator loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()
            
            # ===== GENERATOR UPDATE =====
            optimizer_g.zero_grad()
            
            noise = torch.randn(batch_size, config.latent_dim, device=device)
            fake_images = generator(noise, labels)
            fake_output = discriminator(fake_images, labels)
            real_labels_g = torch.ones(batch_size, 1, device=device)
            loss_g = criterion(fake_output, real_labels_g)
            
            loss_g.backward()
            optimizer_g.step()
            
            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())
            
            pbar.set_postfix({
                'D_Loss': f'{np.mean(d_losses):.4f}',
                'G_Loss': f'{np.mean(g_losses):.4f}'
            })
        
        # Log metrics
        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{avg_d_loss:.6f},{avg_g_loss:.6f}\n")
        
        # ===== SAMPLING =====
        if (epoch + 1) % config.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                fake_images = generator(fixed_noise, fixed_labels)
                fake_images = (fake_images + 1) / 2  # Denormalize
                fake_images = torch.clamp(fake_images, 0, 1)
            
            # Save as grid
            sample_path = sample_dir / f"epoch_{epoch+1:04d}.png"
            
            fig, axes = plt.subplots(
                config.num_classes,
                config.num_sample_images // config.num_classes,
                figsize=(12, 8)
            )
            
            for idx, (img, label) in enumerate(
                zip(fake_images, fixed_labels)
            ):
                row = idx // (config.num_sample_images // config.num_classes)
                col = idx % (config.num_sample_images // config.num_classes)
                ax = axes[row, col]
                
                ax.imshow(img.squeeze().cpu().numpy(), cmap='gray')
                ax.set_title(f"Class {label.item()}", fontsize=8)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(sample_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Samples saved: {sample_path}")
        
        # ===== CHECKPOINT =====
        if (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                epoch + 1,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                checkpoint_dir
            )
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train Conditional GAN"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    
    args = parser.parse_args()
    config_path = args.config
    
    # Load config
    config = load_config(config_path)
    
    # Train
    train_cgan(config)
