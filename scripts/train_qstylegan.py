#!/usr/bin/env python3
"""
Training script for QStyleGAN (Quantum-enhanced StyleGAN2)

Quantum-hybrid style-based GAN for high-quality NEU defect image generation.
Supports class-conditional generation with progressive training.

Usage:
    python scripts/train_qstylegan.py \\
        --config configs/qstylegan_baseline_128.yaml \\
        --data data/NEU_baseline_128 \\
        --output runs/qstylegan_baseline_128
"""

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.qstylegan import QStyleGAN


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path):
    """Setup logging to file and console"""
    log_file = output_dir / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

class NEUDataset(Dataset):
    """NEU steel defects dataset"""
    
    def __init__(self, root_dir: Path, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Class mapping
        self.classes = sorted([
            d for d in (self.root_dir / split).iterdir() if d.is_dir()
        ])
        self.class_to_idx = {c.name: i for i, c in enumerate(self.classes)}
        
        # Build image list
        self.images = []
        for class_dir in self.classes:
            for img_path in class_dir.glob('*.jpg'):
                self.images.append((img_path, self.class_to_idx[class_dir.name]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_data_loader(
    data_dir: Path,
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 4,
    split: str = 'train'
) -> DataLoader:
    """Create data loader for NEU dataset"""
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = NEUDataset(data_dir, split=split, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )


# ============================================================================
# Loss Functions
# ============================================================================

class HingeGANLoss(nn.Module):
    """Hinge loss for GAN training (StyleGAN2 style)"""
    
    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.relu(1.0 + fake).mean() + \
               torch.nn.functional.relu(1.0 - real).mean()
        return loss


class R1Regularization(nn.Module):
    """R1 Gradient penalty (StyleGAN2)"""
    
    def __init__(self, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, real: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        real.backward(torch.ones_like(real), retain_graph=True)
        
        grads = images.grad
        grad_penalty = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
        
        return self.gamma / 2.0 * grad_penalty


# ============================================================================
# Training Loop
# ============================================================================

class QStyleGANTrainer:
    """QStyleGAN trainer"""
    
    def __init__(
        self,
        config: dict,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        output_dir: Path = Path('runs/qstylegan')
    ):
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = setup_logging(self.output_dir)
        self.logger.info(f"🚀 QStyleGAN Training on {device}")
        self.logger.info(f"Config: {json.dumps(config, indent=2)}")
        
        # Model
        self.model = QStyleGAN(
            latent_dim=config['latent_dim'],
            style_dim=config['style_dim'],
            n_classes=config['n_classes'],
            max_resolution=config['image_size'],
            use_quantum=config.get('use_quantum', True),
            n_qubits=config.get('n_qubits', 8)
        ).to(device)
        
        self.logger.info(f"✓ Model initialized")
        
        # Optimizers
        lr_g = config.get('lr_g', 2e-3)
        lr_d = config.get('lr_d', 2e-3)
        beta1 = config.get('beta1', 0.0)
        beta2 = config.get('beta2', 0.99)
        
        self.g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        
        # Loss functions
        self.gan_loss = HingeGANLoss()
        self.r1_reg = R1Regularization(gamma=config.get('r1_gamma', 10.0))
        
        # Training state
        self.global_step = 0
        self.best_fid = float('inf')
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'step': []
        }
    
    def train_step(self, real: torch.Tensor, labels: torch.Tensor) -> dict:
        """Single training step"""
        batch_size = real.shape[0]
        
        # ========== Discriminator Step ==========
        for _ in range(self.config.get('d_iters', 1)):
            # Generate fake images
            z = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
            c = torch.randint(0, self.config['n_classes'], (batch_size,), device=self.device)
            
            with torch.no_grad():
                fake = self.model.generator(z, c)
            
            # Discriminator scores
            real_scores, _ = self.model.discriminator(real, labels)
            fake_scores, _ = self.model.discriminator(fake.detach(), c)
            
            # Hinge loss
            d_loss = self.gan_loss(fake_scores, real_scores)
            
            # R1 regularization (every 16 steps)
            if self.global_step % 16 == 0:
                real.requires_grad_(True)
                real_scores_r1, _ = self.model.discriminator(real, labels)
                r1_loss = self.r1_reg(real_scores_r1.sum(), real)
                d_loss = d_loss + r1_loss
                real.requires_grad_(False)
            
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
        
        # ========== Generator Step ==========
        # Generate new batch
        z = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
        c = torch.randint(0, self.config['n_classes'], (batch_size,), device=self.device)
        
        fake = self.model.generator(z, c)
        
        # Discriminator scores on fake
        fake_scores, _ = self.model.discriminator(fake, c)
        
        # Generator loss (hinge loss, negated)
        g_loss = -fake_scores.mean()
        
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item()
        }
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch"""
        self.model.generator.train()
        self.model.discriminator.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Training step
            losses = self.train_step(images, labels)
            
            total_g_loss += losses['g_loss']
            total_d_loss += losses['d_loss']
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'g_loss': total_g_loss / num_batches,
                'd_loss': total_d_loss / num_batches
            })
            
            # Record history
            self.history['g_loss'].append(losses['g_loss'])
            self.history['d_loss'].append(losses['d_loss'])
            self.history['step'].append(self.global_step)
        
        return {
            'g_loss': total_g_loss / num_batches,
            'd_loss': total_d_loss / num_batches
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'epoch_{epoch:03d}.pt'
        self.model.save(checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / 'best.pt'
            self.model.save(best_path)
        
        self.logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, num_epochs: int):
        """Full training loop"""
        self.logger.info(f"🔥 Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            losses = self.train_epoch(train_loader)
            
            self.logger.info(
                f"G Loss: {losses['g_loss']:.4f} | "
                f"D Loss: {losses['d_loss']:.4f}"
            )
            
            # Save checkpoint
            if epoch % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(epoch)
        
        # Final checkpoint
        self.save_checkpoint(num_epochs, is_best=True)
        
        self.logger.info("✅ Training complete!")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"✓ Training history saved: {history_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train QStyleGAN")
    parser.add_argument('--config', type=str, default='configs/qstylegan_baseline_128.yaml',
                        help='Config file path')
    parser.add_argument('--data', type=str, default='data/NEU_baseline_128',
                        help='Data directory')
    parser.add_argument('--output', type=str, default='runs/qstylegan_baseline_128',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with CLI args
    config['batch_size'] = args.batch_size
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Data loader
    print(f"📁 Loading data from {args.data}")
    train_loader = get_data_loader(
        Path(args.data),
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        split='train'
    )
    print(f"✓ Loaded {len(train_loader.dataset)} training images")
    
    # Trainer
    trainer = QStyleGANTrainer(
        config=config,
        device=device,
        output_dir=Path(args.output)
    )
    
    # Train
    trainer.train(train_loader, num_epochs=args.epochs)


if __name__ == '__main__':
    main()
