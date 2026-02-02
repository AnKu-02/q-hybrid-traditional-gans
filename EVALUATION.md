# GAN Evaluation Framework

Comprehensive evaluation framework for conditional GANs using multiple metrics.

## Overview

The evaluation framework provides end-to-end assessment of GAN quality using:

1. **FID (Fréchet Inception Distance)** - Measures distribution similarity
2. **Label Fidelity** - Measures class conditioning adherence  
3. **Classifier Performance** - Baseline metrics on real data

## Key Concepts

### FID (Fréchet Inception Distance)

**What it measures:** Statistical similarity between real and generated image feature distributions.

**How it works:**
- Extracts features from both real and generated images using InceptionV3
- Fits Gaussian distributions to the features
- Computes distance between the distributions:
  ```
  FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real·Σ_gen)^0.5)
  ```

**Interpretation:**
- Lower FID = better (features more similar)
- Good FID typically < 50 for high-quality generation
- Captures both quality and diversity

**Assumptions:**
- Images should have similar feature distributions regardless of class
- InceptionV3 features are meaningful for this dataset

### Label Fidelity

**What it measures:** Percentage of generated images correctly classified as their conditioning label.

**Why it matters:**
- Ensures generator respects class conditioning
- High fidelity = controlled generation
- Per-class fidelity reveals which classes are easier to generate

**Interpretation:**
- Range: 0.0 to 1.0 (or 0% to 100%)
- Higher is better
- Should be high (>0.8) for well-trained models
- If < 0.5, generator is ignoring class labels

**Assumptions:**
- Classifier trained on real data generalizes to synthetic data
- Classifier is well-trained (validate on real validation set first)
- Label information is consistently encoded in generated images

### Classifier Performance (Real Data)

**What it measures:** Baseline performance of the classifier on real validation images.

**Why it matters:**
- Validates that the classifier is reliable
- Provides context for interpreting label fidelity
- High accuracy (>0.9) indicates trustworthy fidelity scores
- Low accuracy (<0.7) suggests classifier isn't learning features well

**Per-class metrics:**
- Precision: Of predictions for class X, how many were correct?
- Recall: Of true class X images, how many were found?

## Usage

### Basic Evaluation

```bash
# Evaluate CGAN
python scripts/evaluate.py \
    --model cgan \
    --run_name cgan_baseline_128 \
    --config configs/cgan_baseline_128.yaml

# Evaluate QCGAN
python scripts/evaluate.py \
    --model qcgan \
    --run_name qcgan_baseline_128 \
    --config configs/qcgan_baseline_128.yaml
```

### Advanced Options

```bash
python scripts/evaluate.py \
    --model cgan \
    --run_name cgan_baseline_128 \
    --config configs/cgan_baseline_128.yaml \
    --num_images_per_class 100 \
    --classifier_epochs 20 \
    --batch_size 64 \
    --device cuda
```

**Key arguments:**
- `--model`: Model type (cgan or qcgan)
- `--run_name`: Run directory name
- `--config`: Config YAML file
- `--checkpoint`: Custom checkpoint path (optional)
- `--num_images_per_class`: How many images to generate per class (default: 50)
- `--classifier_epochs`: Epochs for training classifier (default: 10)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--device`: cpu or cuda (default: cpu)

## Output Structure

```
runs/<run_name>/evaluation/
├── metrics.json              # Detailed metrics in JSON format
├── metrics.csv               # Tabular metrics in CSV format
└── generated/                # Generated images organized by class
    ├── crazing/
    │   ├── crazing_0000.png
    │   └── ...
    ├── inclusion/
    └── ...
```

### metrics.json Format

```json
{
  "overall_fid": 15.234,
  "per_class_fid": {},
  "classifier_real_accuracy": 0.95,
  "classifier_real_precision": {
    "crazing": 0.94,
    "inclusion": 0.96,
    ...
  },
  "classifier_real_recall": {
    "crazing": 0.95,
    "inclusion": 0.94,
    ...
  },
  "label_fidelity": 0.87,
  "per_class_label_fidelity": {
    "crazing": 0.85,
    "inclusion": 0.90,
    ...
  },
  "num_generated_images": 300,
  "num_real_validation_images": 240
}
```

## Interpretation Guide

### Excellent Results
- **FID < 20**: Generated images are very similar to real images
- **Label Fidelity > 0.9**: Generator strongly respects class conditioning
- **Classifier Accuracy > 0.9**: Baseline classifier is reliable
- **Per-class Fidelity consistent**: Generation quality is balanced

### Good Results
- **FID 20-40**: Generated images are reasonably similar
- **Label Fidelity 0.8-0.9**: Generator mostly respects conditioning
- **Classifier Accuracy 0.8-0.9**: Classifier is reasonably reliable
- **Per-class Fidelity > 0.7**: Most classes generated well

### Warning Signs
- **FID > 60**: Generated images are quite different from real images
- **Label Fidelity < 0.7**: Generator is ignoring class labels
- **Classifier Accuracy < 0.7**: Classifier may not be reliable for fidelity evaluation
- **Large per-class variation**: Some classes are much easier to generate

## Per-Class Analysis

The framework computes separate metrics for each of the 6 defect classes:
- `crazing`: Patterns of fine surface cracks
- `inclusion`: Foreign particles embedded in surface
- `patches`: Surface wear or corrosion spots
- `pitted_surface`: Deep indentations or pitting
- `rolled-in_scale`: Oxide layer rolled into surface
- `scratches`: Linear surface defects

**Use per-class metrics to:**
- Identify which defects are harder to generate
- Spot mode collapse (only generating a few classes well)
- Guide dataset augmentation or training improvements
- Validate that conditioning works for all classes

## Architecture

### Files

**`src/eval/evaluate.py`** - Core evaluation framework
- `EvalConfig`: Configuration dataclass
- `EvalMetrics`: Metrics container
- `ClassificationDataset`: Dataset loader for classification
- `GeneratedImageDataset`: Wrapper for generated tensors
- `FIDCalculator`: FID computation
- `SimpleClassifier`: 4-layer CNN classifier
- `GANEvaluator`: Main orchestrator

**`scripts/evaluate.py`** - Command-line interface
- Argument parsing
- Model loading (CGAN and QCGAN support)
- Full evaluation pipeline
- Results display and saving

### Workflow

1. **Load real data** - Training and validation splits
2. **Generate samples** - Create N images per class from generator
3. **Train classifier** - Train on real training data
4. **Evaluate classifier** - Get baseline accuracy on real validation
5. **Extract features** - Use InceptionV3 on both real and generated
6. **Compute FID** - Compare feature distributions
7. **Compute label fidelity** - Classify generated images, check if correct
8. **Save results** - JSON and CSV formats

## Dependencies

Required packages:
- `torch` - Model loading and generation
- `torchvision` - InceptionV3 for FID
- `numpy` - Numerical computations
- `scipy` - Matrix square root for FID
- `pyyaml` - Config loading
- `tqdm` - Progress bars
- `Pillow` - Image I/O

Optional:
- `torchmetrics` - Alternative FID implementation

## Performance

Typical runtime for full evaluation:
- **Dataset loading**: ~1 second
- **Sample generation** (300 images): ~5-10 seconds (CPU)
- **Classifier training** (10 epochs): ~30-60 seconds (CPU)
- **FID computation**: ~30-60 seconds (CPU)
- **Label fidelity**: ~10-20 seconds (CPU)
- **Total**: ~2-3 minutes (CPU), <1 minute (GPU)

## Customization

### Custom Dataset

Replace `ClassificationDataset` with your own if using different image organization:

```python
# Your dataset should implement:
# - __len__() -> int
# - __getitem__(idx) -> (image_tensor, label_int)
```

### Custom Classifier

Replace `SimpleClassifier` with:
- ResNet for better accuracy
- ViT for more modern architecture
- Pre-trained models from torchvision

### Custom FID Implementation

Can use `torchmetrics.image.fid.FrechetInceptionDistance` if installed:
```bash
pip install torchmetrics
```

## Troubleshooting

**Error: "No checkpoint found"**
- Check that checkpoint file exists in `runs/<run_name>/checkpoints/`
- Use `--checkpoint` to specify path explicitly

**Low classifier accuracy on real data**
- Increase `--classifier_epochs`
- Check that training data is being loaded correctly
- Verify image preprocessing matches training pipeline

**High FID but good label fidelity**
- Generated images are different from real but still recognizable
- Could mean: good class conditioning but poor visual realism
- Try training generator longer

**Low label fidelity but high FID**
- Generated images look real but are misclassified
- Generator may be ignoring class labels
- Check that class conditioning is implemented correctly

## References

- Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
- Lucic, M., et al. (2018). "Are GANs Created Equal? A Large-Scale Empirical Study"
- Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision"
