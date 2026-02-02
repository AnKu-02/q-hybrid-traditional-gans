# StyleGAN2-ADA Implementation Guide

## Overview

This document describes the StyleGAN2-ADA implementation for the NEU defect dataset. StyleGAN2-ADA is an advanced generative model that produces higher quality and more diverse synthetic images compared to standard CGANs.

**Key References:**
- Paper: [Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676)
- Official Code: [NVLabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- Citation: Karras et al., 2020

## Architecture Overview

### Generator (Style-Based)

The StyleGAN2 generator uses a novel **style-based** architecture:

```
Input:
  - Noise vector z: (batch, 512) - random latent codes
  - Class embedding: (batch, num_classes) - class conditioning

Mapping Network:
  - z + class_embed → 8-layer MLP → style vectors w: (batch, 512)
  - Learns disentangled style representation

Constant Initialization:
  - Learnable 4x4 constant: (1, fmap_base, 4, 4)
  - Replaces traditional noise input

Style-Based Synthesis (4 blocks):
  1. 4x4 → 8x8   (upsample + conv + AdaIN with style)
  2. 8x8 → 16x16 (upsample + conv + AdaIN with style)
  3. 16x16 → 64x64 (upsample + conv + AdaIN with style)
  4. 64x64 → 128x128 (upsample + conv + AdaIN with style)

Output: 128x128 grayscale image [-1, 1]

Total Parameters: ~11M
```

### Discriminator (Multi-Scale)

The StyleGAN2 discriminator uses a standard CNN architecture with class conditioning:

```
Input:
  - Image: (batch, 1, 128, 128)
  - Class ID: (batch,)

Feature Extraction (5 blocks):
  1. 128x128 → 64x64  (from_rgb + conv + LeakyReLU + pool)
  2. 64x64 → 32x32   (conv + LeakyReLU + pool)
  3. 32x32 → 16x16   (conv + LeakyReLU + pool)
  4. 16x16 → 8x8     (conv + LeakyReLU + pool)
  5. 8x8 → 4x4       (conv + LeakyReLU + pool)

Classification Head:
  - Final conv: 4x4 feature map → 1x1
  - Flatten → Linear layers → logit (0/1)

Class Conditioning:
  - Class embedding concatenated with features
  - Improves discriminator precision

Total Parameters: ~3.2M
```

### Key Innovation: Adaptive Instance Normalization (AdaIN)

```python
def AdaIN(x, y):
    # x: feature map (batch, C, H, W)
    # y: style vector (batch, C)
    
    # Instance normalization
    mean = x.mean(dim=[2,3], keepdim=True)
    std = x.std(dim=[2,3], keepdim=True)
    x_norm = (x - mean) / std
    
    # Apply style
    return x_norm * y.view(batch, C, 1, 1) + y.view(batch, C, 1, 1)
```

**Benefits:**
- Disentangles style from content
- Enables style mixing (generate coarse and fine features with different styles)
- Better training stability

## Training Configuration

### Hyperparameters

```yaml
# Model
z_dim: 512              # Latent noise dimension
w_dim: 512              # Style dimension (after mapping)
fmap_base: 16384        # Base feature maps
fmap_max: 512           # Cap on feature maps

# Optimization
learning_rate_g: 0.0025 # Generator LR (typically lower than CGAN)
learning_rate_d: 0.0025 # Discriminator LR
betas: [0.0, 0.99]      # Adam momentum (different from CGAN!)
batch_size: 32          # CPU-friendly batch size

# Regularization
use_r1: true            # R1 gradient penalty
r1_gamma: 10.0          # Penalty strength
use_ada: true           # Adaptive augmentation (optional for small datasets)
```

### Loss Functions

**Generator Loss:**
```
L_G = -log(D(G(z, c)))  or  BCE(D(G(z,c)), 1)
```

**Discriminator Loss:**
```
L_D = log(D(x, c)) + log(1 - D(G(z, c), c)) + λ_r1 * R1_penalty
```

**R1 Gradient Penalty:**
```
R1 = (||∇_x D(x,c)||^2)
```

Prevents discriminator from growing unbounded, stabilizes training.

## Training Pipeline

### 1. Data Loading

```python
dataset = NEUDefectDataset(
    metadata_path="data/NEU_baseline_128/metadata.csv",
    image_dir="data/NEU_baseline_128",
    img_size=128
)
```

**Dataset structure:**
- 1,440 training images (240 per class)
- 6 defect classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- Grayscale 128×128 normalized to [-1, 1]

### 2. Model Initialization

```python
generator = StyleGAN2Generator(
    z_dim=512, w_dim=512, img_size=128, num_classes=6
)
discriminator = StyleGAN2Discriminator(
    img_size=128, num_classes=6
)
```

### 3. Training Loop

```
For epoch in 1 to num_epochs:
    For batch in dataloader:
        # Train Discriminator
        real_logits = D(real_images, class_ids)
        fake_logits = D(G(z, class_ids), class_ids)
        
        D_loss = BCE(real_logits, 1) + BCE(fake_logits, 0)
        D_loss += R1_penalty  # Every N iterations
        
        D_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        fake_logits = D(G(z, class_ids), class_ids)
        G_loss = BCE(fake_logits, 1)
        
        G_loss.backward()
        optimizer_g.step()
```

### 4. Checkpointing

Models saved every 5 epochs:
```
runs/stylegan2_ada_baseline_128/
├── checkpoints/
│   ├── epoch_0005.pt
│   ├── epoch_0010.pt
│   ├── epoch_0015.pt
│   └── epoch_0020.pt
├── samples/
│   ├── epoch_0005.png
│   ├── epoch_0010.png
│   ├── epoch_0015.png
│   └── epoch_0020.png
└── logs/
    └── train_log.csv
```

## Usage

### Training from Scratch

```bash
# CPU training (20 epochs)
python scripts/train_stylegan2_ada.py \
    --config configs/stylegan2_ada_baseline_128.yaml

# GPU training (if CUDA available)
python scripts/train_stylegan2_ada.py \
    --config configs/stylegan2_ada_baseline_128.yaml \
    --device cuda

# Override specific parameters
python scripts/train_stylegan2_ada.py \
    --config configs/stylegan2_ada_baseline_128.yaml \
    --epochs 50 \
    --batch-size 64 \
    --lr-g 0.002 \
    --lr-d 0.002
```

### Inference

```bash
# Generate 36 images for class 0
python scripts/inference_stylegan2_ada.py \
    --checkpoint runs/stylegan2_ada_baseline_128/checkpoints/epoch_0020.pt \
    --class-id 0 \
    --num-samples 36 \
    --output generated_crazing.png

# Generate for all classes
python scripts/inference_stylegan2_ada.py \
    --checkpoint runs/stylegan2_ada_baseline_128/checkpoints/epoch_0020.pt \
    --all-classes \
    --output all_defects.png

# Latent space interpolation
python scripts/inference_stylegan2_ada.py \
    --checkpoint runs/stylegan2_ada_baseline_128/checkpoints/epoch_0020.pt \
    --interpolate \
    --class-id 0 \
    --num-steps 10 \
    --output interpolation.png
```

## Comparison: CGAN vs StyleGAN2-ADA

| Aspect | CGAN | StyleGAN2-ADA |
|--------|------|---------------|
| **Architecture** | Noise → Dense → DeConv | Style codes → Mapping → AdaIN |
| **Quality** | Good | Excellent |
| **Diversity** | Moderate | High |
| **Training Stability** | Standard | Improved (R1 + optional ADA) |
| **Disentanglement** | Poor | Excellent |
| **Memory Usage** | Lower | Higher |
| **Training Time** | Faster | Slower |
| **Parameters (G)** | 9.3M | 11M |
| **Parameters (D)** | 2.8M | 3.2M |

**Expected Results:**
- StyleGAN2-ADA: Higher FID scores, better visual quality, more diverse samples
- CGAN: Faster training, lower memory, acceptable quality for defect detection

## Advantages of StyleGAN2-ADA

1. **Style-Based Generation**: Controls image properties at different scales
2. **Adaptive Data Augmentation**: Handles small datasets better than CGAN
3. **Path Length Regularization**: Encourages smooth latent space
4. **R1 Gradient Penalty**: Prevents discriminator collapse
5. **Better Generalization**: Disentangled representations

## Training Tips

1. **Start with CPU**: Estimate training time on CPU before using GPU
2. **Monitor Loss Curves**: D_loss should stay low, G_loss oscillate around 5
3. **Sample Quality**: Check samples/ directory to track image quality
4. **Convergence**: StyleGAN2 needs 20-50 epochs on small datasets
5. **Hyperparameter Tuning**: 
   - Reduce lr if training is unstable
   - Increase r1_gamma for stronger regularization
   - Enable ADA (use_ada: true) for very small datasets

## Output Structure

### Checkpoint Files
```
epoch_XXXX.pt
├── generator (state_dict)
├── discriminator (state_dict)
├── opt_g (Adam state)
└── opt_d (Adam state)
```

### Sample Images
```
epoch_XXXX.png
- Grid of 36 images (6x6)
- 6 samples per class
- Visualizes generation quality at different epochs
```

### Training Logs
```
train_log.csv
Columns: epoch, d_loss, g_loss, r1_penalty
```

## Troubleshooting

### Training is Too Slow
- Reduce batch_size (currently 32, can go to 16)
- Use fewer epochs for testing (currently 20, can reduce to 10)
- Enable GPU if available (device: "cuda")

### Loss Diverges
- Reduce learning rate (currently 0.0025, try 0.0015)
- Increase r1_gamma (currently 10, try 20)
- Ensure training data is properly normalized to [-1, 1]

### Generated Images Look Noisy
- Train for more epochs (currently 20, try 50)
- Check sample_interval is not too high
- Ensure proper loss convergence

### CUDA Memory Error
- Reduce batch_size
- Reduce z_dim (currently 512, can go to 256)
- Use CPU for testing

## References

1. Karras et al. (2020). "Training Generative Adversarial Networks with Limited Data". CVPR.
   - https://arxiv.org/abs/2006.06676

2. Karras et al. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks". CVPR.
   - https://arxiv.org/abs/1812.04948

3. Official Implementation: https://github.com/NVlabs/stylegan2-ada-pytorch

4. DCGAN (base architecture): https://arxiv.org/abs/1511.06434

## Implementation Details

### File Structure
```
train/
├── train_cgan.py              # CGAN training (existing)
└── train_stylegan2_ada.py    # StyleGAN2-ADA training (new)

scripts/
├── train_cgan.py              # CGAN entry point
├── train_stylegan2_ada.py    # StyleGAN2-ADA entry point
└── inference_stylegan2_ada.py # Generate synthetic images

configs/
├── cgan_baseline_128.yaml     # CGAN config (existing)
└── stylegan2_ada_baseline_128.yaml  # StyleGAN2-ADA config (new)

runs/
├── cgan_baseline_128/          # CGAN outputs
└── stylegan2_ada_baseline_128/ # StyleGAN2-ADA outputs
    ├── checkpoints/
    ├── samples/
    └── logs/
```

### Key Classes

**StyleGAN2Generator**
- Maps noise z to style codes w via mapping network
- Uses constant initialization instead of noise input
- Style-based synthesis with AdaIN at each layer
- Class conditioning through embedding

**StyleGAN2Discriminator**
- Multi-scale downsampling (128→8)
- Conditional classification
- R1 gradient penalty support

**AdaIN Module**
- Instance normalization + style application
- Enables style mixing and disentanglement

**NEUDefectDataset**
- Loads metadata CSV
- 1,440 training images across 6 classes
- Normalizes to [-1, 1] range

## Next Steps

1. **Complete CGAN Training**: Wait for epoch 20 completion
2. **Start StyleGAN2-ADA Training**: ~20 minutes after CGAN
3. **Compare Generated Samples**: Visual quality assessment
4. **Evaluate FID Scores**: If compute resources allow
5. **Train Hybrid Detector**: Use both synthetic datasets
6. **Evaluate Detection Performance**: Compare CGAN vs StyleGAN2-ADA augmentation
