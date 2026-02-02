# Conditional GAN Training Guide

## Official Paper & References

**Primary Reference:**
- **"Conditional Generative Adversarial Nets"** (Mirza & Osindski, 2014)
  - Paper: https://arxiv.org/abs/1411.1784
  - TensorLayer Implementation: https://github.com/tensorlayer/dcgan

**Architecture Based On:**
- **DCGAN** (Radford et al., 2015): https://arxiv.org/abs/1511.06434
- Combines DCGAN's stable training with class conditioning

---

## Quick Start

### Installation

First, install required dependencies:

```bash
pip install torch torchvision tqdm pyyaml pillow matplotlib numpy pandas
```

### Training

Run training with a configuration file:

```bash
# Train on baseline dataset (original images)
python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml

# Train on ROI dataset (cropped defect regions)
python scripts/train_cgan.py --config configs/cgan_roi_128.yaml
```

---

## Project Structure

```
├── configs/
│   ├── cgan_baseline_128.yaml    # Config for baseline dataset
│   └── cgan_roi_128.yaml         # Config for ROI dataset
│
├── scripts/
│   └── train_cgan.py              # Entry point script
│
├── train/
│   └── train_cgan.py              # Main training implementation
│
├── runs/                          # Output directory (created during training)
│   ├── cgan_baseline_128/
│   │   ├── checkpoints/           # Model checkpoints
│   │   ├── samples/               # Generated sample grids
│   │   ├── logs/                  # Training logs
│   │   └── config.yaml            # Config used for this run
│   └── cgan_roi_128/
│       ├── checkpoints/
│       ├── samples/
│       ├── logs/
│       └── config.yaml
│
└── data/
    ├── NEU_baseline_128/
    │   └── metadata.csv
    └── NEU_roi_128/
        └── metadata.csv
```

---

## Configuration Guide

Both config files are identical except for `metadata_path` and `run_name`:

### Dataset Configuration
```yaml
metadata_path: "data/NEU_baseline_128/metadata.csv"  # Path to metadata CSV
image_dir: "data/NEU_baseline_128"                    # Base image directory
num_classes: 6                                        # Number of defect classes
img_size: 128                                         # Image size (128x128)
```

### Training Configuration
```yaml
num_epochs: 100              # Number of training epochs
batch_size: 32              # Batch size (adjust for GPU memory)
learning_rate_g: 0.0002     # Generator learning rate
learning_rate_d: 0.0002     # Discriminator learning rate
beta1: 0.5                  # Adam beta1
beta2: 0.999                # Adam beta2
seed: 42                    # Random seed for reproducibility
device: "cuda"              # "cuda" or "cpu"
```

### Model Architecture
```yaml
latent_dim: 100             # Noise vector dimension
base_channels: 64           # Base conv channels (64, 128, 256, 512)
```

### Checkpointing & Logging
```yaml
sample_interval: 5          # Generate samples every N epochs
checkpoint_interval: 10     # Save checkpoint every N epochs
num_sample_images: 36       # Grid of 6x6 = 36 images
```

### Regularization (Optional)
```yaml
use_gradient_penalty: false  # Enable spectral normalization
lambda_gp: 10.0             # Gradient penalty weight
```

### Output
```yaml
run_name: "cgan_baseline_128"         # Run identifier
output_dir: "runs/cgan_baseline_128"  # Output directory
```

---

## Training Output

### Directory Structure

```
runs/cgan_baseline_128/
├── config.yaml              # Configuration file used
├── checkpoints/
│   ├── checkpoint_epoch_0010.pt
│   ├── checkpoint_epoch_0020.pt
│   └── ...
├── samples/
│   ├── epoch_0005.png       # Sample grid (6x6 images, 1 row per class)
│   ├── epoch_0010.png
│   └── ...
└── logs/
    └── train_log.csv        # Training metrics
```

### Training Log (CSV)
```csv
epoch,d_loss,g_loss
1,0.693147,0.693147
2,0.456789,0.534567
3,0.345678,0.456789
...
```

### Sample Grid
- **Rows:** One per defect class (6 rows total)
- **Columns:** 6 generated samples per class
- Generated at intervals specified in config (`sample_interval`)

---

## Model Architecture

### Generator
- **Input:** Noise (100D) + Class embedding (100D) → 200D
- **Architecture:**
  - Dense layer: 200D → 512×8×8
  - 4× Deconvolutional layers (upsampling)
  - Output: 1×128×128 (grayscale image, range [-1, 1])
- **Activation:** ReLU + BatchNorm, Tanh output

### Discriminator
- **Input:** Image (1×128×128) + Class embedding (1×128×128)
- **Architecture:**
  - Concatenate image + class embedding → 2×128×128
  - 4× Convolutional layers (downsampling)
  - Dense layer: 512×8×8 → 1
- **Activation:** LeakyReLU (0.2) + BatchNorm, Sigmoid output

---

## Training Details

### Loss Function
Binary Cross-Entropy (BCE):
```
L_D = BCE(D(real, class), 1) + BCE(D(fake, class), 0)
L_G = BCE(D(fake, class), 1)
```

### Optimization
- **Generator:** Adam, lr=0.0002, β₁=0.5, β₂=0.999
- **Discriminator:** Adam, lr=0.0002, β₁=0.5, β₂=0.999

### Data Normalization
- Images normalized to [-1, 1] range
- Generator output uses Tanh activation
- Discriminator input normalized similarly

---

## Running the Training

### Basic Command
```bash
python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml
```

### Expected Output
```
Device: cuda
======================================================================
Conditional GAN Training
======================================================================
Config: configs/cgan_baseline_128.yaml
Output: runs/cgan_baseline_128
Device: cuda
Epochs: 100
Batch Size: 32
======================================================================

Dataset loaded: 10000 training images
Batches per epoch: 313

Generator parameters: 7,237,953
Discriminator parameters: 7,369,217

Epoch 1/100: 100%|████████| 313/313 [00:45<00:00,  6.96it/s]
D_Loss: 0.6892, G_Loss: 0.6915
...
Samples saved: runs/cgan_baseline_128/samples/epoch_0005.png
Checkpoint saved: runs/cgan_baseline_128/checkpoints/checkpoint_epoch_0010.pt
...

======================================================================
Training completed!
======================================================================
```

---

## Advanced Usage

### Adjusting Hyperparameters

**For faster training (lower quality):**
```yaml
num_epochs: 50
batch_size: 64  # Larger batch
learning_rate_g: 0.0003
learning_rate_d: 0.0003
```

**For better quality (slower training):**
```yaml
num_epochs: 200
batch_size: 16  # Smaller batch
learning_rate_g: 0.0001
learning_rate_d: 0.0001
```

### GPU Memory Management

**If running out of CUDA memory:**
```yaml
batch_size: 16  # Reduce from 32
base_channels: 32  # Reduce from 64
```

**To speed up on GPU:**
```yaml
batch_size: 64  # Increase from 32
base_channels: 128  # Increase from 64
```

### Reproducibility

Set the seed in config:
```yaml
seed: 42
```
This ensures:
- Same random noise initialization
- Same data shuffling
- Same model initialization

---

## Monitoring Training

### Check Loss Convergence
```python
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("runs/cgan_baseline_128/logs/train_log.csv")
plt.plot(log['epoch'], log['d_loss'], label='D Loss')
plt.plot(log['epoch'], log['g_loss'], label='G Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

### Inspect Generated Samples
```bash
# View sample grids
open runs/cgan_baseline_128/samples/epoch_0050.png
```

---

## Loading & Using Trained Models

### Load Checkpoint
```python
import torch
from train.train_cgan import Generator, load_config

# Load config
config = load_config("configs/cgan_baseline_128.yaml")

# Initialize generator
generator = Generator(
    latent_dim=config.latent_dim,
    num_classes=config.num_classes,
    base_channels=config.base_channels,
    img_size=config.img_size
)

# Load weights
checkpoint = torch.load("runs/cgan_baseline_128/checkpoints/checkpoint_epoch_0100.pt")
generator.load_state_dict(checkpoint['generator_state'])
generator.eval()

# Generate samples
with torch.no_grad():
    noise = torch.randn(10, config.latent_dim)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    generated = generator(noise, labels)  # (10, 1, 128, 128)
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution:** Reduce batch size in config
```yaml
batch_size: 16  # from 32
```

### Issue: Discriminator wins (G_Loss → ∞)
**Solution:** Reduce discriminator learning rate
```yaml
learning_rate_d: 0.0001  # from 0.0002
```

### Issue: Mode collapse (same image for all classes)
**Solution:** 
- Check dataset is properly stratified
- Increase latent_dim (100 → 128)
- Verify class labels are correct

### Issue: Poor quality images
**Solution:**
- Train longer (increase num_epochs)
- Use ROI dataset (more focused)
- Verify dataset is loading correctly

---

## Next Steps

1. **Train the model** with baseline config
2. **Monitor samples** in `runs/cgan_baseline_128/samples/`
3. **Evaluate quality** using FID or Inception Score
4. **Fine-tune hyperparameters** based on results
5. **Generate synthetic dataset** for augmentation
6. **Train hybrid detector** combining traditional ML + GAN

---

## References

- Mirza, M., & Osindski, S. (2014). **Conditional Generative Adversarial Nets**. https://arxiv.org/abs/1411.1784
- Radford, A., Metz, L., & Chintala, S. (2015). **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**. https://arxiv.org/abs/1511.06434
- Karras, T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J., & Aila, T. (2020). **Training Generative Adversarial Networks with Limited Data**. https://arxiv.org/abs/2006.06676
