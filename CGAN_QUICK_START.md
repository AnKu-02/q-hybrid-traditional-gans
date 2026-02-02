# Quick Start Reference Card

## Installation
```bash
pip install -r requirements_cgan.txt
```

## Training Commands

### Baseline Dataset (Original Images)
```bash
python scripts/train_cgan.py --config configs/cgan_baseline_128.yaml
```

### ROI Dataset (Defect-Focused Crops)
```bash
python scripts/train_cgan.py --config configs/cgan_roi_128.yaml
```

## Expected Output
- **Training Time:** ~2 hours (GPU) for 100 epochs
- **Output Location:** `runs/cgan_baseline_128/` or `runs/cgan_roi_128/`
- **Checkpoints:** `runs/*/checkpoints/checkpoint_epoch_*.pt`
- **Samples:** `runs/*/samples/epoch_*.png` (6×6 grid)
- **Logs:** `runs/*/logs/train_log.csv`

## After Training

### Generate Images & Export
```bash
python inference_cgan.py
```

Outputs:
- `generated_samples_grid.png` - 6×6 grid of all classes
- `real_vs_generated.png` - Comparison with real images
- `training_curves.png` - Loss visualization
- `data/NEU_roi_128_generated_from_cgan_baseline_128/` - Exported dataset

## Configuration Quick Guide

Edit `configs/cgan_baseline_128.yaml` to customize:

```yaml
# Dataset
metadata_path: "data/NEU_baseline_128/metadata.csv"
image_dir: "data/NEU_baseline_128"

# Training
num_epochs: 100
batch_size: 32  # Reduce if OOM
learning_rate_g: 0.0002
learning_rate_d: 0.0002

# Model
latent_dim: 100
base_channels: 64  # Reduce if OOM

# Sampling & Checkpointing
sample_interval: 5  # Save samples every N epochs
checkpoint_interval: 10  # Save checkpoint every N epochs

# Output
run_name: "cgan_baseline_128"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce `batch_size` to 16 or 8 |
| **Discriminator winning** | Reduce `learning_rate_d` to 0.0001 |
| **Generator winning** | Reduce `learning_rate_g` to 0.0001 |
| **Mode collapse** | Train longer, verify dataset balance |
| **Poor quality** | Train for more epochs (200+) |
| **Config file not found** | Verify path is relative to current directory |

## Monitor Training

1. Check real-time output in terminal
2. View sample grids in `runs/*/samples/`
3. Plot metrics: `tail -f runs/*/logs/train_log.csv`

## Papers & References

- **CGAN:** https://arxiv.org/abs/1411.1784
- **DCGAN:** https://arxiv.org/abs/1511.06434
- **StyleGAN2 (for future improvements):** https://arxiv.org/abs/2006.06676

## File Structure

```
├── train/train_cgan.py                    # Core implementation
├── scripts/train_cgan.py                  # Entry point
├── configs/
│   ├── cgan_baseline_128.yaml
│   └── cgan_roi_128.yaml
├── inference_cgan.py                      # Generate & visualize
├── requirements_cgan.txt
├── CGAN_TRAINING_GUIDE.md                 # Full guide
├── CGAN_IMPLEMENTATION_SUMMARY.md         # Technical details
└── runs/                                  # Output (created)
```

## Next Steps After Training

1. **Review Samples**
   - Open `runs/cgan_baseline_128/samples/epoch_0100.png`
   - Assess visual quality per class

2. **Export Generated Dataset**
   - Run `python inference_cgan.py`
   - Check `data/NEU_roi_128_generated_from_*/`

3. **Train Hybrid Detector**
   - Combine real + synthetic images
   - Train SVM or Random Forest classifier

4. **Evaluate & Compare**
   - Test on held-out validation set
   - Compare with baseline augmentation

---

**Questions?** See full guide: `CGAN_TRAINING_GUIDE.md`
