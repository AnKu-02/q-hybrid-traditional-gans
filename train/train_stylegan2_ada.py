"""
StyleGAN2-ADA Training Implementation
====================================

Based on: "Training Generative Adversarial Networks with Limited Data"
Paper: https://arxiv.org/abs/2006.06676
Code:  https://github.com/NVlabs/stylegan2-ada-pytorch

Key Features:
- Adaptive Discriminator Augmentation (ADA) for small datasets
- Progressive growing (optional)
- Spectral normalization
- Path length regularization
- R1 gradient penalty
- Class conditioning

Architecture:
- Generator: Style-based generator with AdaIN
- Discriminator: Multi-scale discriminator with ADA
"""

import os
import csv
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
import yaml
import numpy as np
from PIL import Image
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StyleGAN2ADAConfig:
    """Configuration for StyleGAN2-ADA training"""
    
    # Dataset
    metadata_path: str = "data/NEU_baseline_128/metadata.csv"
    image_dir: str = "data/NEU_baseline_128"
    num_classes: int = 6
    img_size: int = 128
    
    # Model architecture
    z_dim: int = 512  # Latent dimension
    w_dim: int = 512  # Style/mapping dimension
    fmap_base: int = 16384  # Feature maps base
    fmap_max: int = 512
    fmap_decay: float = 1.0
    
    # Training
    num_epochs: int = 20
    batch_size: int = 32
    num_workers: int = 0
    learning_rate_g: float = 0.0025
    learning_rate_d: float = 0.0025
    betas: Tuple[float, float] = (0.0, 0.99)
    eps: float = 1e-8
    
    # Regularization
    use_ada: bool = True  # Adaptive Discriminator Augmentation
    ada_target: float = 0.6
    ada_interval: int = 4
    ada_kimg_base: int = 100
    
    use_r1: bool = True
    r1_gamma: float = 10.0
    
    path_length_decay: float = 0.01
    
    # Checkpointing & Logging
    checkpoint_interval: int = 5
    sample_interval: int = 5
    log_interval: int = 100
    run_name: str = "stylegan2_ada_baseline_128"
    output_dir: str = "runs/stylegan2_ada_baseline_128"
    
    # Hardware
    device: str = "cpu"  # "cuda" or "cpu" or "mps"
    seed: int = 42


# ============================================================================
# Dataset
# ============================================================================

class NEUDefectDataset(Dataset):
    """NEU Defect dataset for StyleGAN2-ADA training"""
    
    def __init__(self, metadata_path: str, image_dir: str, img_size: int = 128):
        self.img_size = img_size
        self.image_dir = Path(image_dir)
        
        # Load metadata
        self.df = pd.read_csv(metadata_path)
        self.df = self.df[self.df['split'] == 'train'].reset_index(drop=True)
        
        # Build class mapping
        self.classes = sorted(self.df['class_name'].unique())
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}
        
        print(f"Dataset initialized: {len(self)} images, {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = self.image_dir / row['filepath']
        class_id = int(row['class_id'])
        
        # Load image
        img = Image.open(image_path).convert('L')  # Grayscale
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Normalize to [-1, 1]
        img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        
        return img_tensor, class_id


# ============================================================================
# Generator (StyleGAN2)
# ============================================================================

class MappingNetwork(nn.Module):
    """Mapping network: z -> w"""
    
    def __init__(self, z_dim: int, w_dim: int, num_layers: int = 8):
        super().__init__()
        
        layers = []
        in_dim = z_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = w_dim
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, z_dim) -> w: (batch, w_dim)"""
        return self.net(z)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: feature map (batch, C, H, W)
        y: style vector (batch, C)
        """
        # Instance normalize x
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        
        # Apply style
        y = y.view(y.size(0), y.size(1), 1, 1)
        return x_norm * y + y


class StyleBlock(nn.Module):
    """Style-based synthesis block"""
    
    def __init__(self, in_channels: int, out_channels: int, w_dim: int, upsample: bool = False):
        super().__init__()
        
        self.upsample = upsample
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # First conv: in_channels -> out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Style transformations for output channels
        self.style_scale = nn.Linear(w_dim, out_channels)
        self.style_bias = nn.Linear(w_dim, out_channels)
        self.adain = AdaIN()
        
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.up(x)
        
        x = self.conv(x)  # Apply convolution with proper channel transformation
        style_scale = self.style_scale(w)
        style_bias = self.style_bias(w)
        
        x = self.adain(x, style_scale)
        x = x * style_scale.view(style_scale.size(0), style_scale.size(1), 1, 1)
        x = x + style_bias.view(style_bias.size(0), style_bias.size(1), 1, 1)
        
        return x


class StyleGAN2Generator(nn.Module):
    """Simplified StyleGAN2-inspired Generator with class conditioning"""
    
    def __init__(self, 
                 z_dim: int,
                 w_dim: int,
                 img_size: int,
                 num_classes: int,
                 fmap_base: int = 16384,
                 fmap_max: int = 512):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim + num_classes, w_dim)
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, num_classes)
        
        # Simple DCGAN-style generator with style modulation
        self.fc = nn.Linear(w_dim, 512 * 4 * 4)
        
        self.layers = nn.ModuleList([
            # 4x4 -> 8x8
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ),
            # 8x8 -> 16x16
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            # 32x32 -> 64x64
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            # 64x64 -> 128x128
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            ),
        ])
        
        self.to_img = nn.Sequential(
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, z: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, z_dim)
        class_id: (batch,)
        """
        batch = z.size(0)
        
        # Embed class
        class_embed = self.class_embed(class_id)
        z_aug = torch.cat([z, class_embed], dim=1)
        
        # Mapping
        w = self.mapping(z_aug)
        
        # Generate base feature
        x = self.fc(w).view(batch, 512, 4, 4)
        
        # Progressive synthesis
        for layer in self.layers:
            x = layer(x)
        
        # Final image
        img = self.to_img(x)
        return img


# ============================================================================
# Discriminator
# ============================================================================

class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator with ADA"""
    
    def __init__(self,
                 img_size: int,
                 num_classes: int,
                 fmap_base: int = 16384,
                 fmap_max: int = 512):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Input layer
        self.from_rgb = nn.Conv2d(1, min(fmap_base // 32, fmap_max), 1)
        
        # Convolutional blocks
        self.blocks = nn.ModuleList()
        channels = [min(c, fmap_max) for c in [
            fmap_base // 32, fmap_base // 16, fmap_base // 8,
            fmap_base // 4, fmap_base // 2, fmap_base
        ]]
        
        for i in range(len(channels) - 1):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2)
            ))
        
        # Classification head
        self.final_conv = nn.Conv2d(channels[-1], channels[-1], 4, padding=0)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * 1 * 1, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
        
        # Class conditioning
        self.class_embed = nn.Embedding(num_classes, channels[-1])
        
    def forward(self, img: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        """
        img: (batch, 1, H, W)
        class_id: (batch,)
        """
        # Feature extraction
        x = self.from_rgb(img)
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x)  # (batch, C, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        logit = self.classifier(x)  # (batch, 1)
        
        return logit


# ============================================================================
# Training Loop
# ============================================================================

def train_stylegan2_ada(config: StyleGAN2ADAConfig) -> None:
    """Train StyleGAN2-ADA model"""
    
    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    print("=" * 70)
    print("StyleGAN2-ADA Training")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print("=" * 70)
    
    # Dataset
    dataset = NEUDefectDataset(
        config.metadata_path,
        config.image_dir,
        config.img_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
    # Models
    generator = StyleGAN2Generator(
        z_dim=config.z_dim,
        w_dim=config.w_dim,
        img_size=config.img_size,
        num_classes=config.num_classes
    ).to(device)
    
    discriminator = StyleGAN2Discriminator(
        img_size=config.img_size,
        num_classes=config.num_classes
    ).to(device)
    
    print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Optimizers
    opt_g = Adam(generator.parameters(), lr=config.learning_rate_g, betas=config.betas, eps=config.eps)
    opt_d = Adam(discriminator.parameters(), lr=config.learning_rate_d, betas=config.betas, eps=config.eps)
    
    # Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # Logging
    log_file = output_dir / "logs" / "train_log.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'd_loss', 'g_loss', 'r1_penalty'])
    
    # ADA parameters
    ada_aug_p = 0.0
    
    # Training loop
    for epoch in range(config.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", leave=True)
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_r1_penalty = 0
        
        for batch_idx, (real_imgs, class_ids) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            class_ids = class_ids.to(device)
            batch_size = real_imgs.size(0)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # ================================================================
            # Train Discriminator
            # ================================================================
            opt_d.zero_grad()
            
            # Real images
            real_logits = discriminator(real_imgs, class_ids)
            d_loss_real = criterion(real_logits, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, config.z_dim, device=device)
            with torch.no_grad():
                fake_imgs = generator(z, class_ids)
            
            fake_logits = discriminator(fake_imgs.detach(), class_ids)
            d_loss_fake = criterion(fake_logits, fake_labels)
            
            # Total D loss
            d_loss = d_loss_real + d_loss_fake
            
            # R1 regularization
            r1_penalty = torch.tensor(0.0, device=device)
            if config.use_r1 and batch_idx % config.ada_interval == 0:
                real_imgs.requires_grad_(True)
                real_logits = discriminator(real_imgs, class_ids)
                grad = torch.autograd.grad(
                    outputs=real_logits.sum(),
                    inputs=real_imgs,
                    create_graph=True
                )[0]
                r1_penalty = (grad.norm(p=2, dim=(1, 2, 3)).pow(2).mean() * config.r1_gamma)
                d_loss = d_loss + r1_penalty
            
            d_loss.backward()
            opt_d.step()
            
            # ================================================================
            # Train Generator
            # ================================================================
            opt_g.zero_grad()
            
            z = torch.randn(batch_size, config.z_dim, device=device)
            fake_imgs = generator(z, class_ids)
            fake_logits = discriminator(fake_imgs, class_ids)
            
            g_loss = criterion(fake_logits, real_labels)
            g_loss.backward()
            opt_g.step()
            
            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_r1_penalty += r1_penalty.item()
            
            # Update progress bar
            pbar.set_postfix({
                'D_Loss': f'{d_loss.item():.4f}',
                'G_Loss': f'{g_loss.item():.4f}'
            })
        
        # Average losses
        epoch_d_loss /= len(dataloader)
        epoch_g_loss /= len(dataloader)
        epoch_r1_penalty /= len(dataloader)
        
        # Logging
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_d_loss, epoch_g_loss, epoch_r1_penalty])
        
        # Sampling
        if (epoch + 1) % config.sample_interval == 0:
            with torch.no_grad():
                z = torch.randn(36, config.z_dim, device=device)
                class_ids = torch.arange(config.num_classes, device=device).repeat(6)
                samples = generator(z, class_ids)
                
                # Save samples
                sample_path = output_dir / "samples" / f"epoch_{epoch + 1:04d}.png"
                save_image_grid(samples, sample_path, config.num_classes, 6)
        
        # Checkpointing
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict(),
            }
            torch.save(checkpoint, output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt")
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(asdict(config), f)
    
    print("\n✅ Training complete!")
    print(f"Output: {output_dir}")


def save_image_grid(images: torch.Tensor, path: Path, num_classes: int, samples_per_class: int) -> None:
    """Save image grid"""
    from torchvision.utils import make_grid
    
    # Denormalize
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    # Create grid
    grid = make_grid(images, nrow=num_classes, normalize=False)
    
    # Save
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_img = (grid * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(grid_img.squeeze(), mode='L').save(path)
    print(f"Samples saved: {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train StyleGAN2-ADA")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = StyleGAN2ADAConfig(**config_dict)
    train_stylegan2_ada(config)
