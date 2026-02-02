# Conditional GAN Architecture Details

## Overview

Conditional GAN (CGAN) extends standard GAN by conditioning both generator and discriminator on class labels. This allows us to generate specific types of defects on demand.

**Key Paper:** "Conditional Generative Adversarial Nets" (Mirza & Osinski, 2014)
- https://arxiv.org/abs/1411.1784

**Architecture Base:** DCGAN (Radford et al., 2015)
- https://arxiv.org/abs/1511.06434

---

## Generator Architecture

### Purpose
Generate 128×128 grayscale images of industrial defects for a specified class.

### Input
- **Noise vector:** `z` ~ N(0, I), shape: (batch_size, 100)
- **Class label:** Integer in [0, 5], shape: (batch_size,)

### Processing Pipeline

```
Input Processing:
─────────────────────────────────────────────────────────────────
    Noise z (100,)                     Class label (scalar)
         ↓                                    ↓
    [No change]                     Embedding Layer (6→100)
         ↓                                    ↓
    Noise (100,)              Label Embedding (100,)
    ╰──────────────────────────────────────────╯
              Concatenate
                   ↓
         Combined (200,)
─────────────────────────────────────────────────────────────────

Dense Layer (FC):
─────────────────────────────────────────────────────────────────
     Combined (200,)
           ↓
    Linear: 200 → 512×8×8 = 32,768
           ↓
    Reshape: (batch, 512, 8, 8)
    
    Parameters: 200 × 32,768 + 32,768 = 6.5M
─────────────────────────────────────────────────────────────────

Deconvolutional Layers (Upsampling):
─────────────────────────────────────────────────────────────────

Layer 1: DeConv 512→256
    Input:  (batch, 512, 8, 8)
    Conv:   kernel=4, stride=2, padding=1
    BN:     Batch Normalization
    Act:    ReLU
    Output: (batch, 256, 16, 16)
    Params: 512×256×4×4 + 256 = 2.1M
    
Layer 2: DeConv 256→128
    Input:  (batch, 256, 16, 16)
    Conv:   kernel=4, stride=2, padding=1
    BN:     Batch Normalization
    Act:    ReLU
    Output: (batch, 128, 32, 32)
    Params: 256×128×4×4 + 128 = 0.5M
    
Layer 3: DeConv 128→64
    Input:  (batch, 128, 32, 32)
    Conv:   kernel=4, stride=2, padding=1
    BN:     Batch Normalization
    Act:    ReLU
    Output: (batch, 64, 64, 64)
    Params: 128×64×4×4 + 64 = 131K
    
Layer 4: DeConv 64→1
    Input:  (batch, 64, 64, 64)
    Conv:   kernel=4, stride=2, padding=1
    Act:    Tanh (output range [-1, 1])
    Output: (batch, 1, 128, 128)
    Params: 64×1×4×4 + 1 = 1K

─────────────────────────────────────────────────────────────────
Total Parameters: ~7.2M
Computation: O(H×W×C) per layer, cumulative upsampling ops
─────────────────────────────────────────────────────────────────
```

### Output
- **Generated Image:** (batch_size, 1, 128, 128)
- **Value Range:** [-1, 1] (normalized for training stability)
- **Activation:** Tanh (preferred over Sigmoid for symmetric range)

### Key Design Choices

1. **Embedding instead of one-hot:** More compact (100D vs 6D)
2. **Concatenate noise + label:** Combines randomness with class information
3. **Batch Normalization:** Stabilizes training, prevents internal covariate shift
4. **Gradual upsampling:** 8×8 → 16×16 → 32×32 → 64×64 → 128×128
5. **Tanh activation:** Output range [-1, 1] matches input normalization

---

## Discriminator Architecture

### Purpose
Classify whether an image is real or generated, and verify it matches the given class label.

### Input
- **Image:** (batch_size, 1, 128, 128), range [-1, 1]
- **Class label:** Integer in [0, 5], shape: (batch_size,)

### Processing Pipeline

```
Input Processing:
─────────────────────────────────────────────────────────────────
    Image (1, 128, 128)              Class label (scalar)
         ↓                                    ↓
    [No change]                Embedding Layer (6→16,384)
         ↓                                    ↓
    Image (1, 128, 128)         Reshape (1, 128, 128)
    ╰──────────────────────────────────────────╯
         Concatenate (channel-wise)
                   ↓
         Combined (2, 128, 128)
─────────────────────────────────────────────────────────────────

Convolutional Layers (Downsampling):
─────────────────────────────────────────────────────────────────

Layer 1: Conv 2→64
    Input:  (batch, 2, 128, 128)
    Conv:   kernel=4, stride=2, padding=1
    Act:    LeakyReLU(0.2)  [NOT BatchNorm in first layer - DCGAN style]
    Output: (batch, 64, 64, 64)
    Params: 2×64×4×4 = 2K
    
Layer 2: Conv 64→128
    Input:  (batch, 64, 64, 64)
    Conv:   kernel=4, stride=2, padding=1
    BN:     Batch Normalization
    Act:    LeakyReLU(0.2)
    Output: (batch, 128, 32, 32)
    Params: 64×128×4×4 + 128 = 131K
    
Layer 3: Conv 128→256
    Input:  (batch, 128, 32, 32)
    Conv:   kernel=4, stride=2, padding=1
    BN:     Batch Normalization
    Act:    LeakyReLU(0.2)
    Output: (batch, 256, 16, 16)
    Params: 128×256×4×4 + 256 = 0.5M
    
Layer 4: Conv 256→512
    Input:  (batch, 256, 16, 16)
    Conv:   kernel=4, stride=2, padding=1
    BN:     Batch Normalization
    Act:    LeakyReLU(0.2)
    Output: (batch, 512, 8, 8)
    Params: 256×512×4×4 + 512 = 2.1M

─────────────────────────────────────────────────────────────────

Classification:
─────────────────────────────────────────────────────────────────
    Feature Map (512, 8, 8)
           ↓
    Flatten: 512×8×8 = 32,768
           ↓
    Dense: 32,768 → 1
           ↓
    Sigmoid: Output ∈ [0, 1]
           ↓
    Probability (real=1, fake=0)

    Params: 32,768 × 1 + 1 = 32.8K

─────────────────────────────────────────────────────────────────
Total Parameters: ~7.4M
Computation: O(H×W×C) per layer, cumulative downsampling ops
─────────────────────────────────────────────────────────────────
```

### Output
- **Classification:** (batch_size, 1)
- **Value Range:** [0, 1] (probability real image)
- **Activation:** Sigmoid (binary classification)

### Key Design Choices

1. **Channel-wise concatenation:** Preserves spatial structure
2. **No BatchNorm in first layer:** Helps discriminator learn from true data distribution
3. **LeakyReLU:** Allows small negative gradients (helps with gradient flow)
4. **Gradual downsampling:** 128×128 → 64×64 → 32×32 → 16×16 → 8×8
5. **Sigmoid output:** Binary classification probability

---

## Training Loop

### Forward Pass - Discriminator

```python
# Real images
real_output = discriminator(real_images, labels)
loss_d_real = BCE(real_output, ones)  # Target: 1 (real)

# Fake images
fake_images = generator(noise, labels)
fake_output = discriminator(fake_images.detach(), labels)
loss_d_fake = BCE(fake_output, zeros)  # Target: 0 (fake)

# Total loss
loss_d = loss_d_real + loss_d_fake
```

### Forward Pass - Generator

```python
# Generate and classify
fake_images = generator(noise, labels)
fake_output = discriminator(fake_images, labels)
loss_g = BCE(fake_output, ones)  # Target: 1 (fool discriminator)
```

### Loss Functions

**Binary Cross-Entropy (BCE):**
```
BCE(p, y) = -[y * log(p) + (1-y) * log(1-p)]
```

**Discriminator Loss:**
```
L_D = E[BCE(D(real, c), 1)] + E[BCE(D(fake, c), 0)]
    = E[-log D(real, c)] + E[-log(1 - D(fake, c))]
```

**Generator Loss:**
```
L_G = E[BCE(D(fake, c), 1)]
    = E[-log D(fake, c)]
```

### Optimization

**Adam Optimizer:**
```
θ_g ← θ_g - lr_g ∇L_G
θ_d ← θ_d - lr_d ∇L_D

Parameters:
  learning_rate = 0.0002
  beta1 = 0.5  (momentum)
  beta2 = 0.999  (variance)
```

---

## Data Flow Diagram

```
GENERATOR:
─────────────────────────────────────────────────────────────
Random Noise (100,)
    ↓
[Class Embedding Layer]
    ↓
Class Embedding (100,)
    ↓
Concatenate → (200,)
    ↓
Dense FC Layer → (512×8×8)
    ↓
Reshape → (512, 8, 8)
    ↓
[DeConv + BN + ReLU] → (256, 16, 16)
    ↓
[DeConv + BN + ReLU] → (128, 32, 32)
    ↓
[DeConv + BN + ReLU] → (64, 64, 64)
    ↓
[DeConv + Tanh] → (1, 128, 128)
    ↓
Generated Image ∈ [-1, 1]


DISCRIMINATOR:
─────────────────────────────────────────────────────────────
Real/Fake Image (1, 128, 128)
    ↓
[Class Embedding Layer]
    ↓
Class Embedding (1, 128, 128)
    ↓
Concatenate → (2, 128, 128)
    ↓
[Conv + LeakyReLU] → (64, 64, 64)
    ↓
[Conv + BN + LeakyReLU] → (128, 32, 32)
    ↓
[Conv + BN + LeakyReLU] → (256, 16, 16)
    ↓
[Conv + BN + LeakyReLU] → (512, 8, 8)
    ↓
Flatten → (32,768)
    ↓
Dense + Sigmoid → (1)
    ↓
Probability ∈ [0, 1]
```

---

## Batch Norm & Activation Functions

### Batch Normalization
```
Purpose: Normalize layer inputs to reduce internal covariate shift

In Generator:
- Applied to all DeConv layers except output
- Helps with training stability
- Prevents saturation in early layers

In Discriminator:
- Applied to all Conv layers except first
- First layer learns from raw distribution
- Helps with adversarial stability
```

### LeakyReLU in Discriminator
```
LeakyReLU(x) = max(0.2x, x)

Advantages:
- Allows small negative gradients
- Prevents "dead neurons"
- Better gradient flow during backprop
- Standard for GAN discriminators
```

---

## Parameter Counts

| Component | Layers | Parameters |
|-----------|--------|-----------|
| Generator | Dense + 4 DeConv | 7,237,953 |
| Discriminator | 4 Conv + Dense | 7,369,217 |
| **Total** | **8 layers** | **14,607,170** |

---

## Training Dynamics

### Good Training Behavior
```
Epoch 1:   D_Loss ≈ 0.693, G_Loss ≈ 0.693  (Random baseline)
Epoch 10:  D_Loss ≈ 0.500, G_Loss ≈ 0.500  (Learning begins)
Epoch 50:  D_Loss ≈ 0.300, G_Loss ≈ 0.400  (Convergence)
Epoch 100: D_Loss ≈ 0.250, G_Loss ≈ 0.350  (Stable training)
```

### What Can Go Wrong

**Mode Collapse:** Generator produces same image for all classes
- Symptom: G_Loss decreases, D_Loss increases
- Cause: Generator exploits D weakness
- Fix: More diverse training data, increase latent dim

**Discriminator Wins:** G_Loss → ∞, D_Loss → 0
- Symptom: Generator can't fool discriminator
- Cause: D too strong
- Fix: Reduce lr_d, increase base_channels for G

**Unstable Training:** Large loss oscillations
- Symptom: Losses jump around
- Cause: Learning rates too high
- Fix: Reduce learning rates by 50%

---

## References

1. **Conditional GAN Paper**
   - Mirza, M., & Osinski, S. (2014)
   - https://arxiv.org/abs/1411.1784

2. **DCGAN Paper**
   - Radford, A., Metz, L., & Chintala, S. (2015)
   - https://arxiv.org/abs/1511.06434

3. **Spectral Normalization (Optional Enhancement)**
   - Miyato, T., et al. (2018)
   - https://arxiv.org/abs/1802.05957

4. **StyleGAN2 Improvements**
   - Karras, T., et al. (2020)
   - https://arxiv.org/abs/2006.06676
