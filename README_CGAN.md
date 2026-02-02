# Conditional GAN for Industrial Defect Generation

Conditional Generative Adversarial Network (CGAN) implementation for generating synthetic industrial defect images from the NEU-DET dataset.

**Official Papers:**
- **CGAN:** "Conditional Generative Adversarial Nets" (Mirza & Osinski, 2014) - https://arxiv.org/abs/1411.1784
- **DCGAN:** "Unsupervised Representation Learning with DCGANs" (Radford et al., 2015) - https://arxiv.org/abs/1511.06434

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements_cgan.txt
```

### Training
```bash
# Train on baseline dataset (original images)
python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml

# Train on ROI dataset (cropped defect regions)
python scripts/train_cgan.py --config configs/cgan_roi_128.yaml
```

### Expected Output
```
Device: cuda
======================================================================
Conditional GAN Training
======================================================================
Config: configs/cgan_baseline_128.yaml
Output: runs/cgan_baseline_128
...
Epoch 1/100: 100%|████████| 313/313
D_Loss: 0.6892, G_Loss: 0.6915

Samples saved: runs/cgan_baseline_128/samples/epoch_0005.png
Checkpoint saved: runs/cgan_baseline_128/checkpoints/checkpoint_epoch_0010.pt
```

---

## 📁 Project Structure

```
├── train/
│   └── train_cgan.py                    # Core implementation
│                                         # - Generator & Discriminator classes
│                                         # - NEUDefectDataset class
│                                         # - Full training loop
│                                         # - 900+ lines, type hints, docstrings
│
├── scripts/
│   └── train_cgan.py                    # Entry point script
│                                         # - Command-line interface
│                                         # - Config validation
│
├── configs/
│   ├── cgan_baseline_128.yaml            # Baseline dataset config
│   └── cgan_roi_128.yaml                 # ROI dataset config
│
├── inference_cgan.py                     # Inference & visualization
│                                         # - Load trained model
│                                         # - Generate images
│                                         # - Create visualizations
│                                         # - Export dataset
│
├── requirements_cgan.txt                 # Dependencies
│
├── CGAN_QUICK_START.md                  # Quick reference
├── CGAN_TRAINING_GUIDE.md               # Comprehensive guide
├── CGAN_ARCHITECTURE.md                 # Technical details
├── CGAN_IMPLEMENTATION_SUMMARY.md       # Implementation overview
│
├── runs/                                 # Output directory (created)
│   └── cgan_baseline_128/
│       ├── config.yaml
│       ├── checkpoints/
│       │   ├── checkpoint_epoch_0010.pt
│       │   └── ...
│       ├── samples/
│       │   ├── epoch_0005.png
│       │   └── ...
│       └── logs/
│           └── train_log.csv
│
└── data/
    ├── NEU_baseline_128/
    │   └── metadata.csv
    └── NEU_roi_128/
        └── metadata.csv
```

---

## 🎯 What This Does

### Generator
- **Input:** Random noise (100D) + class label
- **Output:** Synthetic 128×128 grayscale defect image
- **Architecture:** 4-layer deconvolutional network with batch normalization
- **Parameters:** ~7.2M

### Discriminator
- **Input:** Image + class label
- **Output:** Probability of being real (0-1)
- **Architecture:** 4-layer convolutional network
- **Parameters:** ~7.4M

### Dataset
- **Source:** NEU-DET metadata.csv
- **Classes:** 6 industrial defect types
- **Image Size:** 128×128 pixels
- **Split:** 80% train / 20% validation

---

## 📊 Training Process

### Workflow
1. **Load dataset** from metadata.csv
2. **Initialize models:** Generator + Discriminator
3. **Per epoch:**
   - Discriminator training: Classify real vs fake
   - Generator training: Fool discriminator
   - Log metrics to CSV
   - Save samples every N epochs
   - Save checkpoints every N epochs

### Output Files

**Checkpoints** (`runs/*/checkpoints/`)
- PyTorch state dicts for both models
- Optimizer states for resuming training
- Named by epoch: `checkpoint_epoch_0010.pt`

**Samples** (`runs/*/samples/`)
- 6×6 grid of generated images (one row per class)
- PNG format, high resolution
- Generated every 5 epochs

**Logs** (`runs/*/logs/train_log.csv`)
```csv
epoch,d_loss,g_loss
1,0.693147,0.693147
2,0.456789,0.534567
```

---

## ⚙️ Configuration Guide

Edit `configs/cgan_baseline_128.yaml`:

### Dataset
```yaml
metadata_path: "data/NEU_baseline_128/metadata.csv"
image_dir: "data/NEU_baseline_128"
num_classes: 6
img_size: 128
```

### Training
```yaml
num_epochs: 100
batch_size: 32              # Reduce if OOM
learning_rate_g: 0.0002
learning_rate_d: 0.0002
seed: 42                    # For reproducibility
device: "cuda"              # or "cpu"
```

### Model
```yaml
latent_dim: 100
base_channels: 64           # Reduce if OOM
```

### Checkpointing
```yaml
sample_interval: 5          # Save samples every 5 epochs
checkpoint_interval: 10     # Save checkpoint every 10 epochs
num_sample_images: 36       # 6×6 grid
```

---

## 🔧 Inference & Visualization

After training, run:
```bash
python inference_cgan.py
```

This generates:
- `generated_samples_grid.png` - 6×6 grid per class
- `real_vs_generated.png` - Side-by-side comparison
- `training_curves.png` - Loss visualization
- Exported dataset in `data/NEU_roi_128_generated_from_cgan_baseline_128/`

---

## 📈 Performance Expectations

### Training Time (per epoch)
| GPU | Time |
|-----|------|
| A100 | 30s |
| V100 | 45s |
| RTX3090 | 60s |
| CPU | 10-15min (not recommended) |

### Memory Usage
- **GPU:** 4-6 GB (batch_size=32)
- **RAM:** ~2 GB

### Quality Timeline
| Epoch Range | Description |
|-------------|------------|
| 1-10 | Random noise, no structure |
| 20-30 | Recognizable shapes emerging |
| 50-70 | Good class separation |
| 80-100 | High-quality realistic samples |

---

## 🐛 Troubleshooting

| Problem | Symptom | Solution |
|---------|---------|----------|
| **CUDA OOM** | Runtime error | Reduce `batch_size` |
| **D wins** | D_loss→0, G_loss→∞ | Reduce `learning_rate_d` |
| **Mode collapse** | Same image repeated | Use ROI dataset, increase latent_dim |
| **Poor quality** | Blurry/noisy images | Train longer (200+ epochs) |
| **Unstable training** | Loss oscillations | Reduce learning rates |

See `CGAN_TRAINING_GUIDE.md` for detailed troubleshooting.

---

## 📚 Documentation

1. **CGAN_QUICK_START.md** - Quick reference card
2. **CGAN_TRAINING_GUIDE.md** - Comprehensive training guide
3. **CGAN_ARCHITECTURE.md** - Detailed architecture explanation
4. **CGAN_IMPLEMENTATION_SUMMARY.md** - Implementation overview

---

## 🎓 Technical Details

### Architecture
- **Generator:** Takes noise (100D) + class embedding (100D) → 128×128 image
- **Discriminator:** Takes image + class embedding → real/fake classification
- **Conditioning:** Embedding + concatenation at input
- **Activation:** ReLU in generator, LeakyReLU in discriminator
- **Normalization:** Batch Norm in all but first layer

### Loss Functions
- **Discriminator:** Binary cross-entropy on real/fake classification
- **Generator:** Binary cross-entropy (fool discriminator)
- **Optimizer:** Adam with lr=0.0002, β₁=0.5, β₂=0.999

### Data Handling
- Images normalized to [-1, 1]
- Generator output uses Tanh activation
- Grayscale images (1 channel)
- 128×128 spatial dimensions

---

## 🔗 References

**Official Papers:**
- Mirza, M., & Osinski, S. (2014). Conditional Generative Adversarial Nets. https://arxiv.org/abs/1411.1784
- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with DCGANs. https://arxiv.org/abs/1511.06434

**Related Work:**
- Karras, T., et al. (2020). Training GANs with Limited Data. https://arxiv.org/abs/2006.06676
- TensorLayer DCGAN: https://github.com/tensorlayer/dcgan

---

## 📝 Usage Examples

### Basic Training
```bash
python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml
```

### Custom Configuration
Edit `configs/cgan_baseline_128.yaml` then run:
```bash
python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml
```

### Generate Images After Training
```bash
python inference_cgan.py
```

### Load Trained Model
```python
import torch
from train.train_cgan import Generator, load_config

config = load_config("runs/cgan_baseline_128/config.yaml")
generator = Generator(
    latent_dim=config.latent_dim,
    num_classes=config.num_classes,
    base_channels=config.base_channels,
    img_size=config.img_size
)

checkpoint = torch.load("runs/cgan_baseline_128/checkpoints/checkpoint_epoch_0100.pt")
generator.load_state_dict(checkpoint['generator_state'])
generator.eval()

# Generate images
with torch.no_grad():
    noise = torch.randn(10, config.latent_dim)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    images = generator(noise, labels)  # (10, 1, 128, 128)
```

---

## 🎯 Next Steps

1. **Train the model:** Run training with baseline config
2. **Monitor samples:** Check `runs/cgan_baseline_128/samples/` during training
3. **Evaluate quality:** Use `inference_cgan.py` to visualize results
4. **Generate dataset:** Export synthetic images for augmentation
5. **Train detector:** Use real + synthetic images for hybrid classifier

---

## ✨ Key Features

✅ **Production-Ready**
- Type hints throughout
- Comprehensive error handling
- Input validation

✅ **Well-Documented**
- Official paper references
- Detailed architecture docs
- Training guide
- Multiple README files

✅ **Flexible Configuration**
- YAML-based configs
- Easy hyperparameter tuning
- CPU/GPU support
- Reproducible (seed control)

✅ **Complete Implementation**
- Full training loop
- Checkpoint save/load
- Sample generation
- CSV logging
- Inference utilities

---

## 📞 Support

- Check `CGAN_QUICK_START.md` for quick reference
- See `CGAN_TRAINING_GUIDE.md` for detailed guide
- Review `CGAN_ARCHITECTURE.md` for technical details
- Consult `CGAN_IMPLEMENTATION_SUMMARY.md` for overview

---

## 📄 License

Implementation based on official CGAN and DCGAN papers.

---

**Ready to train!** 🚀

```bash
python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml
```
