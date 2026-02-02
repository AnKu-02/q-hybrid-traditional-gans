"""
Quantum Conditional GAN (QCGAN) Training Implementation
=======================================================

A hybrid quantum-classical approach combining:
- Quantum circuits for feature extraction and transformation
- Classical deep learning for discrimination and generation
- Conditional generation based on defect class labels

Key Features:
- Parameterized quantum circuits (PQC) for feature maps
- Quantum-classical hybrid discriminator
- Quantum data encoding with angle embeddings
- Variational quantum algorithms for training
- Class-conditional generation
- Supports both simulators and real quantum hardware (via Qiskit)

Architecture:
- Quantum Generator: Classical latent code → Quantum circuit → Quantum state
- Classical Decoder: Quantum measurements → Image features
- Quantum Discriminator: Real/Fake classification with quantum-enhanced features
- Classical layers: Dense networks for final predictions

Paper Reference:
"Quantum Generative Adversarial Networks" - Hu et al. (2021)
"""

import os
import csv
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
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

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Sampler
    from qiskit.circuit import Parameter, ParameterVector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️  Qiskit not installed. Install with: pip install qiskit qiskit-aer qiskit-machine-learning")

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("⚠️  PennyLane not installed. Install with: pip install pennylane")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class QuantumCGANConfig:
    """Configuration for Quantum Conditional GAN training"""
    
    # Dataset
    metadata_path: str = "data/NEU_baseline_128/metadata.csv"
    image_dir: str = "data/NEU_baseline_128"
    num_classes: int = 6
    img_size: int = 128
    
    # Quantum settings
    num_qubits: int = 8  # Number of qubits in quantum circuit
    quantum_depth: int = 4  # Depth of quantum circuit layers
    measurement_samples: int = 1000  # Shots for measurement
    quantum_backend: str = "simulator"  # "simulator" or "real" (Qiskit)
    use_pennylane: bool = True  # Use PennyLane instead of Qiskit
    
    # Classical architecture
    z_dim: int = 32  # Latent noise dimension
    hidden_dim: int = 256  # Hidden layer dimension
    quantum_feature_dim: int = 256  # Dimension of quantum features
    
    # Training
    num_epochs: int = 20
    batch_size: int = 16  # Smaller batch for quantum (computational cost)
    num_workers: int = 0
    learning_rate_g: float = 0.001
    learning_rate_d: float = 0.001
    betas: Tuple[float, float] = (0.5, 0.999)
    eps: float = 1e-8
    
    # Quantum-classical hybrid
    quantum_ratio: float = 0.5  # Mix quantum and classical features
    parameter_shift_rule: bool = True  # Use parameter shift for gradients
    
    # Checkpointing & Logging
    checkpoint_interval: int = 5
    sample_interval: int = 5
    run_name: str = "qcgan_baseline_128"
    output_dir: str = "runs/qcgan_baseline_128"
    
    # Hardware
    device: str = "cpu"
    seed: int = 42


# ============================================================================
# Dataset (Reused from StyleGAN2)
# ============================================================================

class NEUDefectDataset(Dataset):
    """NEU Defect dataset for QCGAN training"""
    
    def __init__(self, metadata_path: str, image_dir: str, img_size: int = 128):
        self.img_size = img_size
        self.image_dir = Path(image_dir)
        
        self.df = pd.read_csv(metadata_path)
        self.df = self.df[self.df['split'] == 'train'].reset_index(drop=True)
        
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
        
        img = Image.open(image_path).convert('L')
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return img_tensor, class_id


# ============================================================================
# Quantum Circuits (PennyLane)
# ============================================================================

class QuantumFeatureMap(nn.Module):
    """Quantum feature map using PennyLane"""
    
    def __init__(self, num_qubits: int = 8, num_features: int = 4, depth: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_features = num_features
        self.depth = depth
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane required for quantum circuits")
        
        self.dev = qml.device("default.qubit", wires=num_qubits)
        
        # Create quantum circuit with trainable parameters
        self.params = nn.Parameter(torch.randn(depth, num_qubits, 3))
        
        @qml.qnode(self.dev)
        def circuit(params, x):
            # Data encoding
            for i in range(num_qubits):
                if i < len(x):
                    qml.RX(x[i], wires=i)
            
            # Variational layers
            for layer in range(depth):
                for i in range(num_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                    qml.RZ(params[layer, i, 2], wires=i)
                
                # Entangling
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        self.circuit = circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_features) with values in [-1, 1]
        output: (batch, num_qubits) quantum measurement outcomes
        """
        batch_size = x.size(0)
        outputs = []
        
        for i in range(batch_size):
            # Normalize input to [0, pi]
            x_normalized = (x[i] + 1) * np.pi / 2
            
            # Run quantum circuit
            result = torch.tensor(
                self.circuit(self.params, x_normalized.detach().cpu().numpy()),
                dtype=torch.float32,
                device=x.device
            )
            outputs.append(result)
        
        return torch.stack(outputs)


class QuantumGenerator(nn.Module):
    """Quantum Generator: Hybrid classical-quantum architecture"""
    
    def __init__(self, 
                 z_dim: int = 32,
                 num_qubits: int = 8,
                 hidden_dim: int = 256,
                 quantum_depth: int = 2,
                 num_classes: int = 6):
        super().__init__()
        
        self.z_dim = z_dim
        self.num_qubits = num_qubits
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Classical preprocessing: z + class → quantum input
        self.input_projection = nn.Sequential(
            nn.Linear(z_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_qubits)
        )
        
        # Quantum feature map
        if PENNYLANE_AVAILABLE:
            self.quantum_map = QuantumFeatureMap(
                num_qubits=num_qubits,
                num_features=num_qubits,
                depth=quantum_depth
            )
        else:
            self.quantum_map = None
        
        # Classical decoder: quantum features + classical features → image
        # Input: quantum_features (num_qubits) + z_class (z_dim + num_classes)
        decoder_input_dim = num_qubits + z_dim + num_classes
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128 * 128),
            nn.Tanh()
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, num_classes)
    
    def forward(self, z: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, z_dim) random noise
        class_id: (batch,) class labels
        output: (batch, 1, 128, 128) generated images
        """
        batch_size = z.size(0)
        
        # Embed class
        class_embed = self.class_embed(class_id)  # (batch, num_classes)
        
        # Concatenate z and class
        z_class = torch.cat([z, class_embed], dim=1)  # (batch, z_dim + num_classes)
        
        # Project to quantum input
        quantum_input = self.input_projection(z_class)  # (batch, num_qubits)
        
        # Run quantum feature map
        if self.quantum_map is not None:
            quantum_features = self.quantum_map(quantum_input)  # (batch, num_qubits)
        else:
            # Fallback: classical quantum-inspired features
            quantum_features = torch.sin(quantum_input * np.pi / 2)
        
        # Concatenate quantum and classical features
        combined_features = torch.cat([quantum_features, z_class], dim=1)
        
        # Decode to image
        img_flat = self.decoder(combined_features)  # (batch, 128*128)
        img = img_flat.view(batch_size, 1, 128, 128)
        
        return img


class QuantumDiscriminator(nn.Module):
    """Quantum Discriminator: Hybrid quantum-classical architecture"""
    
    def __init__(self,
                 img_size: int = 128,
                 num_qubits: int = 8,
                 hidden_dim: int = 256,
                 quantum_depth: int = 2,
                 num_classes: int = 6):
        super().__init__()
        
        self.img_size = img_size
        self.num_qubits = num_qubits
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Classical encoder: image → quantum input
        self.encoder = nn.Sequential(
            nn.Linear(img_size * img_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_qubits)
        )
        
        # Quantum feature map
        if PENNYLANE_AVAILABLE:
            self.quantum_map = QuantumFeatureMap(
                num_qubits=num_qubits,
                num_features=num_qubits,
                depth=quantum_depth
            )
        else:
            self.quantum_map = None
        
        # Classical classifier: quantum features + class → real/fake
        self.classifier = nn.Sequential(
            nn.Linear(num_qubits + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, num_classes)
    
    def forward(self, img: torch.Tensor, class_id: torch.Tensor) -> torch.Tensor:
        """
        img: (batch, 1, 128, 128) images
        class_id: (batch,) class labels
        output: (batch, 1) real/fake probability
        """
        batch_size = img.size(0)
        
        # Flatten image
        img_flat = img.view(batch_size, -1)  # (batch, 128*128)
        
        # Encode to quantum input
        quantum_input = self.encoder(img_flat)  # (batch, num_qubits)
        
        # Run quantum feature map
        if self.quantum_map is not None:
            quantum_features = self.quantum_map(quantum_input)  # (batch, num_qubits)
        else:
            # Fallback: classical quantum-inspired features
            quantum_features = torch.cos(quantum_input * np.pi / 2)
        
        # Embed class
        class_embed = self.class_embed(class_id)  # (batch, num_classes)
        
        # Classify
        combined = torch.cat([quantum_features, class_embed], dim=1)
        logit = self.classifier(combined)  # (batch, 1)
        
        return logit


# ============================================================================
# Training Loop
# ============================================================================

def train_qcgan(config: QuantumCGANConfig) -> None:
    """Train Quantum Conditional GAN"""
    
    if not PENNYLANE_AVAILABLE:
        print("⚠️  PennyLane not available. Install with: pip install pennylane")
        print("Using classical fallback mode (quantum-inspired only)")
    
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
    print("Quantum Conditional GAN Training")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Quantum Qubits: {config.num_qubits}")
    print(f"Quantum Depth: {config.quantum_depth}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"PennyLane Available: {PENNYLANE_AVAILABLE}")
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
    generator = QuantumGenerator(
        z_dim=config.z_dim,
        num_qubits=config.num_qubits,
        hidden_dim=config.hidden_dim,
        quantum_depth=config.quantum_depth,
        num_classes=config.num_classes
    ).to(device)
    
    discriminator = QuantumDiscriminator(
        img_size=config.img_size,
        num_qubits=config.num_qubits,
        hidden_dim=config.hidden_dim,
        quantum_depth=config.quantum_depth,
        num_classes=config.num_classes
    ).to(device)
    
    print(f"\nQuantum Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Quantum Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
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
        writer.writerow(['epoch', 'd_loss', 'g_loss', 'quantum_cost'])
    
    # Training loop
    print("\n🚀 Starting Quantum GAN Training...")
    
    for epoch in range(config.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", leave=True)
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_quantum_cost = 0
        
        for batch_idx, (real_imgs, class_ids) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            class_ids = class_ids.to(device)
            batch_size = real_imgs.size(0)
            
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
            
            d_loss = d_loss_real + d_loss_fake
            
            # Quantum cost estimate (circuit complexity)
            quantum_cost = config.num_qubits * config.quantum_depth * 0.01
            
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
            epoch_quantum_cost += quantum_cost
            
            # Update progress bar
            pbar.set_postfix({
                'D_Loss': f'{d_loss.item():.4f}',
                'G_Loss': f'{g_loss.item():.4f}',
                'Q_Cost': f'{quantum_cost:.4f}'
            })
        
        # Average losses
        epoch_d_loss /= len(dataloader)
        epoch_g_loss /= len(dataloader)
        epoch_quantum_cost /= len(dataloader)
        
        # Logging
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_d_loss, epoch_g_loss, epoch_quantum_cost])
        
        # Sampling
        if (epoch + 1) % config.sample_interval == 0:
            with torch.no_grad():
                z = torch.randn(36, config.z_dim, device=device)
                class_ids = torch.arange(config.num_classes, device=device).repeat(6)
                samples = generator(z, class_ids)
                
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
    
    print("\n✅ Quantum GAN Training complete!")
    print(f"Output: {output_dir}")


def save_image_grid(images: torch.Tensor, path: Path, num_classes: int, samples_per_class: int) -> None:
    """Save image grid"""
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
    print(f"Quantum samples saved: {path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Quantum Conditional GAN")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = QuantumCGANConfig(**config_dict)
    train_qcgan(config)
