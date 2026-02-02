"""
Inference & Visualization Notebook for Conditional GAN

This notebook demonstrates how to:
1. Load trained checkpoints
2. Generate synthetic images
3. Visualize results
4. Export generated dataset
"""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

# ============================================================================
# SETUP
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME = "cgan_baseline_128"
RUN_DIR = Path(f"runs/{RUN_NAME}")
CHECKPOINT_EPOCH = 100  # Which checkpoint to load

CLASSES = ["crazing", "inclusion", "patches", 
           "pitted_surface", "rolled-in_scale", "scratches"]

print(f"Device: {DEVICE}")
print(f"Run: {RUN_NAME}")
print(f"Checkpoint: Epoch {CHECKPOINT_EPOCH}")


# ============================================================================
# LOAD TRAINED GENERATOR
# ============================================================================

# Import generator (assuming train_cgan.py is accessible)
import sys
sys.path.insert(0, str(Path.cwd()))

from train.train_cgan import Generator, load_config

# Load config
config = load_config(str(RUN_DIR / "config.yaml"))

# Initialize generator
generator = Generator(
    latent_dim=config.latent_dim,
    num_classes=config.num_classes,
    base_channels=config.base_channels,
    img_size=config.img_size
).to(DEVICE)

# Load checkpoint
checkpoint_path = (
    RUN_DIR / "checkpoints" / f"checkpoint_epoch_{CHECKPOINT_EPOCH:04d}.pt"
)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
generator.load_state_dict(checkpoint['generator_state'])
generator.eval()

print(f"✓ Loaded checkpoint: {checkpoint_path}")
print(f"✓ Generator has {sum(p.numel() for p in generator.parameters()):,} parameters")


# ============================================================================
# GENERATE IMAGES
# ============================================================================

def generate_images(
    generator: nn.Module,
    num_per_class: int = 5,
    latent_dim: int = 100,
    num_classes: int = 6
) -> dict:
    """
    Generate images for all classes.
    
    Returns:
        Dict mapping class_id -> tensor of generated images (N, 1, H, W)
    """
    generator.eval()
    generated = {}
    
    with torch.no_grad():
        for class_id in range(num_classes):
            noise = torch.randn(
                num_per_class, latent_dim,
                device=DEVICE
            )
            labels = torch.full(
                (num_per_class,),
                class_id,
                dtype=torch.long,
                device=DEVICE
            )
            
            images = generator(noise, labels)
            # Denormalize
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
            
            generated[class_id] = images.cpu()
    
    return generated


# Generate 5 samples per class
print("\nGenerating samples...")
generated_images = generate_images(
    generator,
    num_per_class=5,
    latent_dim=config.latent_dim,
    num_classes=config.num_classes
)

print(f"✓ Generated {sum(len(v) for v in generated_images.values())} images")


# ============================================================================
# VISUALIZATION: Grid Plot
# ============================================================================

def plot_generated_grid(
    generated: dict,
    classes: list,
    title: str = "Generated Defect Images"
):
    """Plot generated images as grid (one row per class)."""
    num_classes = len(generated)
    num_per_class = generated[0].size(0)
    
    fig, axes = plt.subplots(
        num_classes, num_per_class,
        figsize=(12, 10)
    )
    
    for class_id in range(num_classes):
        for sample_id in range(num_per_class):
            ax = axes[class_id, sample_id]
            
            img = generated[class_id][sample_id].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            
            if sample_id == 0:
                ax.set_ylabel(classes[class_id], fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            ax.set_title('')
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# Plot and show
fig = plot_generated_grid(generated_images, CLASSES)
plt.savefig("generated_samples_grid.png", dpi=150, bbox_inches='tight')
print("✓ Saved: generated_samples_grid.png")
plt.show()


# ============================================================================
# COMPARISON: Real vs Generated
# ============================================================================

def plot_real_vs_generated(
    real_dir: Path,
    generated: dict,
    classes: list,
    num_samples: int = 3
):
    """Compare real images with generated images."""
    fig, axes = plt.subplots(
        len(classes), num_samples * 2,
        figsize=(14, 12)
    )
    
    for class_id, class_name in enumerate(classes):
        # Real images
        real_path = real_dir / class_name
        real_files = sorted(list(real_path.glob("*.png")))[:num_samples]
        
        for sample_id, real_file in enumerate(real_files):
            ax = axes[class_id, sample_id]
            
            real_img = Image.open(real_file).convert('L')
            ax.imshow(np.array(real_img), cmap='gray')
            ax.set_title("Real", fontsize=9)
            ax.axis('off')
        
        # Generated images
        for sample_id in range(num_samples):
            ax = axes[class_id, num_samples + sample_id]
            
            gen_img = generated[class_id][sample_id].squeeze().numpy()
            ax.imshow(gen_img, cmap='gray')
            ax.set_title("Generated", fontsize=9, color='blue')
            ax.axis('off')
        
        # Class label
        axes[class_id, 0].set_ylabel(class_name, fontsize=10, fontweight='bold')
    
    fig.suptitle(f"Real vs Generated ({num_samples} samples per class)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# Compare with training dataset
real_dir = Path("data/NEU_roi_128_augmented/train")
if real_dir.exists():
    fig = plot_real_vs_generated(real_dir, generated_images, CLASSES, num_samples=3)
    plt.savefig("real_vs_generated.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: real_vs_generated.png")
    plt.show()


# ============================================================================
# EXPORT: Save Generated Dataset
# ============================================================================

def export_generated_dataset(
    generated: dict,
    output_dir: Path,
    classes: list,
    num_images_per_class: int = 1000
):
    """Export generated images as dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    for class_id, class_name in enumerate(classes):
        class_dir = output_dir / "train" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate enough images
        num_batches = (num_images_per_class + generated[class_id].size(0) - 1) // generated[class_id].size(0)
        
        image_count = 0
        for batch_idx in range(num_batches):
            # Generate new batch
            with torch.no_grad():
                noise = torch.randn(
                    generated[class_id].size(0), config.latent_dim,
                    device=DEVICE
                )
                labels = torch.full(
                    (generated[class_id].size(0),),
                    class_id,
                    dtype=torch.long,
                    device=DEVICE
                )
                
                batch_images = generator(noise, labels)
                batch_images = (batch_images + 1) / 2
                batch_images = torch.clamp(batch_images, 0, 1)
            
            for img_idx, img_tensor in enumerate(batch_images):
                if image_count >= num_images_per_class:
                    break
                
                # Save image
                img_array = (img_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_array, mode='L')
                
                image_name = f"{class_name}_{image_count:05d}.png"
                image_path = class_dir / image_name
                img_pil.save(image_path)
                
                # Log metadata
                metadata.append({
                    'image_path': f"train/{class_name}/{image_name}",
                    'class': class_name,
                    'split': 'train',
                    'source': 'generated'
                })
                
                image_count += 1
        
        print(f"✓ Generated {image_count} {class_name} images")
    
    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(output_dir / "metadata.csv", index=False)
    print(f"✓ Saved metadata: {output_dir / 'metadata.csv'}")
    
    return meta_df


# Export generated dataset
print("\nExporting generated dataset...")
export_dir = Path(f"data/NEU_roi_128_generated_from_{RUN_NAME}")
meta_df = export_generated_dataset(
    generated_images,
    export_dir,
    CLASSES,
    num_images_per_class=100  # 100 per class for demo
)

print(f"✓ Exported to: {export_dir}")
print(f"✓ Total images: {len(meta_df)}")


# ============================================================================
# ANALYSIS: Training Progress
# ============================================================================

def plot_training_curves():
    """Plot training losses over time."""
    log_file = RUN_DIR / "logs" / "train_log.csv"
    
    if not log_file.exists():
        print(f"⚠ Log file not found: {log_file}")
        return
    
    df = pd.read_csv(log_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Discriminator loss
    ax1.plot(df['epoch'], df['d_loss'], label='D Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Discriminator Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Generator loss
    ax2.plot(df['epoch'], df['g_loss'], label='G Loss', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Generator Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.suptitle(f"Training Curves - {RUN_NAME}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: training_curves.png")
    plt.show()


plot_training_curves()


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("INFERENCE COMPLETE")
print("="*70)
print(f"\nGenerated Files:")
print(f"  • generated_samples_grid.png - Grid of generated images")
print(f"  • real_vs_generated.png - Comparison with real images")
print(f"  • training_curves.png - Training loss curves")
print(f"\nExported Dataset:")
print(f"  • {export_dir}")
print(f"  • {len(meta_df)} total generated images")
print(f"\nNext Steps:")
print(f"  1. Review generated samples in generated_samples_grid.png")
print(f"  2. Compare quality with real_vs_generated.png")
print(f"  3. Use generated dataset for augmentation")
print("="*70 + "\n")
