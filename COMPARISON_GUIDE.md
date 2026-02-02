# GAN Comparison Script Guide

## Overview

`scripts/compare_runs.py` is a thesis-ready utility for comparing evaluation metrics between two conditional GAN runs (typically baseline vs. ROI dataset variants).

## Quick Start

```bash
# Compare CGAN baseline vs ROI runs
python scripts/compare_runs.py \
  --baseline runs/cgan_baseline_128 \
  --roi runs/cgan_roi_128 \
  --output runs/cgan_comparison.csv \
  --report runs/cgan_comparison_report.md

# Compare QCGAN variants
python scripts/compare_runs.py \
  --baseline runs/qcgan_baseline_128 \
  --roi runs/qcgan_roi_128
```

## Inputs

### Required Arguments

- **`--baseline`**: Path to baseline run directory
  - Must contain `evaluation/metrics.json`
  - Example: `runs/cgan_baseline_128`

- **`--roi`**: Path to ROI run directory
  - Must contain `evaluation/metrics.json`
  - Example: `runs/cgan_roi_128`

### Optional Arguments

- **`--output`**: Output CSV path
  - Default: `runs/comparison_results.csv`
  - File contains structured metrics for publication tables

- **`--report`**: Output markdown report path
  - Default: `runs/comparison_report.md`
  - File contains full analysis and interpretation guidance

## Outputs

### 1. Comparison CSV (`comparison_results.csv`)

**Purpose**: Publication-ready metrics table

**Columns**:
- `run_name` - Name of the run (from directory name)
- `dataset_variant` - Auto-detected variant (Baseline/ROI)
- `FID` - Fréchet Inception Distance score
- `label_fidelity` - Label fidelity percentage
- `classifier_val_accuracy` - Classifier accuracy on real validation data
- `per_class_fidelity_json` - JSON object with per-class fidelity scores

**Example**:
```
run_name,dataset_variant,FID,label_fidelity,classifier_val_accuracy,per_class_fidelity_json
cgan_baseline_128,Baseline,5.8840,0.5000,0.7083,{"crazing":1.0,"inclusion":0.0,...}
cgan_roi_128,ROI,12.3456,0.6667,0.7500,{"crazing":0.8,"inclusion":0.5,...}
```

### 2. Comparison Report (`comparison_report.md`)

**Purpose**: Detailed thesis-ready analysis

**Sections**:
1. **Overview** - Run names and dataset information
2. **Summary Statistics** - Side-by-side metrics with percentage differences
3. **Detailed Analysis** - Per-metric interpretation and insights
4. **Per-Class Fidelity Analysis** - Class-by-class breakdown
5. **Key Differences** - Dataset characteristics and trade-offs
6. **Interpretation Guidance** - When to use each variant
7. **Limitations** - Important caveats and assumptions
8. **Recommendations** - For publication and further analysis

## Metrics Explained

### FID Score
- **Range**: 0-∞ (lower is better)
- **Meaning**: Feature distribution distance between real and generated images
- **Good Range**: 5-20 (excellent < 10, good 10-30, fair 30-50)
- **How Computed**: InceptionV3 feature extraction → Gaussian statistics → Fréchet distance

### Label Fidelity
- **Range**: 0.0-1.0 (higher is better)
- **Meaning**: Percentage of generated images correctly classified as their conditioning label
- **Good Range**: 0.85-0.95+
- **How Computed**: Run trained classifier on generated images, compute accuracy

### Classifier Accuracy (Real Data)
- **Range**: 0.0-1.0 (higher is better)
- **Meaning**: Baseline classifier performance on real validation images
- **Purpose**: Indicates how reliable the label fidelity measurement is
- **Interpretation**: 
  - >0.85: Very reliable fidelity measurements
  - 0.70-0.85: Reasonably reliable
  - <0.70: Take fidelity scores with caution

## Example Workflow

### Step 1: Train Models
```bash
# Train baseline model
python scripts/train.py --config configs/cgan_baseline_128.yaml

# Train ROI model (if available)
python scripts/train.py --config configs/cgan_roi_128.yaml
```

### Step 2: Evaluate Both
```bash
# Evaluate baseline
python scripts/evaluate.py --model cgan --run_name cgan_baseline_128 \
  --config configs/cgan_baseline_128.yaml --num_images_per_class 50

# Evaluate ROI
python scripts/evaluate.py --model cgan --run_name cgan_roi_128 \
  --config configs/cgan_roi_128.yaml --num_images_per_class 50
```

### Step 3: Generate Comparison
```bash
python scripts/compare_runs.py \
  --baseline runs/cgan_baseline_128 \
  --roi runs/cgan_roi_128 \
  --output runs/cgan_baseline_vs_roi.csv \
  --report runs/cgan_baseline_vs_roi_analysis.md
```

### Step 4: Use for Thesis
- **Metrics Table**: Copy CSV to Excel/Google Sheets for publication
- **Analysis**: Include markdown report in thesis appendix or supplementary materials
- **Figures**: Generate comparison charts from CSV data

## Interpretation Examples

### High FID, Low Fidelity (BAD)
```
FID: 45.2, Fidelity: 0.15
→ Generated images poor quality & don't respect class labels
→ Model needs retraining
```

### Low FID, High Fidelity (GOOD)
```
FID: 8.5, Fidelity: 0.92
→ Generated images realistic & respect class conditioning
→ Production ready
```

### Low FID, Low Fidelity (MODE COLLAPSE)
```
FID: 3.2, Fidelity: 0.25
→ Realistic images but ignoring class information
→ Check training convergence, increase λ_cond in loss
```

### High FID, High Fidelity (DIVERSITY ISSUE)
```
FID: 38.1, Fidelity: 0.88
→ Respects classes but images are blurry/repetitive
→ Increase noise diversity, check discriminator strength
```

## Limitations

⚠️ **Important for Thesis**

1. **FID Stability**
   - Small sample sizes (default 30 per class) create variance
   - Recommended: 100+ per class for publication
   - Use `--num_images_per_class 100` when evaluating

2. **Classifier Dependency**
   - Fidelity depends on classifier quality
   - Quick training (5 epochs) may underestimate fidelity
   - Train for full convergence for final results

3. **Single Checkpoint**
   - Evaluates only one checkpoint per model
   - May not represent steady-state performance
   - Compare multiple checkpoints for robustness

4. **Dataset Size**
   - Comparison reflects specific training conditions
   - May not generalize to other architectures
   - Note hyperparameters and dataset details in thesis

## Customization

### Change Detection Logic
```python
# In compare_runs.py, modify get_dataset_variant()
def get_dataset_variant(run_name: str) -> str:
    if "my_variant" in run_name.lower():
        return "My Variant"
    # ... rest of logic
```

### Add Custom Metrics
```python
# In generate_markdown_report(), add:
custom_metric = roi_metrics.get('my_custom_metric', 0)
report += f"### 4. My Custom Metric\n{custom_metric}\n"
```

### Change Report Format
```python
# Modify the report template in generate_markdown_report()
# Can add figures, equations, references, etc.
```

## Publication Tips

### For Conference Papers (Limited Space)
```markdown
## Results

We compared models on [baseline/ROI] dataset variants using FID and label fidelity:

| Model | FID ↓ | Fidelity ↑ |
|-------|-------|-----------|
| CGAN  | 5.88  | 0.50      |
| QCGAN | 19.90 | 0.17      |

CGAN achieves 70% lower FID and 3× higher fidelity...
```

### For Journal Papers (Detailed Analysis)
- Include full markdown report as supplementary material
- Reference CSV in data availability statement
- Add interpretation guidance section

### For Thesis
- Include comparison metrics in results chapter
- Add interpretation guidance as methodology note
- Reference limitations in future work section
- Include full report in appendix

## Troubleshooting

### "No checkpoint found in runs/..."
- The run directory must have completed training
- Check `runs/<run_name>/checkpoints/` exists with `.pt` files

### "metrics.json not found"
- The run must have completed evaluation
- Run `python scripts/evaluate.py --run_name <run_name>` first

### Comparison shows all values as "N/A"
- Check metrics.json is valid JSON: `python -m json.tool runs/<run_name>/evaluation/metrics.json`
- Verify evaluation completed successfully

### CSV values are truncated
- CSV is designed for readability, not for full precision
- Use `per_class_fidelity_json` column for detailed values
- Or parse metrics.json directly for exact values

## Advanced Usage

### Batch Comparison
```bash
# Compare all CGAN variants
for variant in baseline roi; do
  python scripts/compare_runs.py \
    --baseline runs/cgan_baseline_128 \
    --roi runs/cgan_${variant}_128 \
    --output runs/cgan_vs_${variant}.csv
done
```

### Extract Metrics Programmatically
```python
import json
from pathlib import Path

metrics = json.load(open('runs/cgan_baseline_128/evaluation/metrics.json'))
print(f"FID: {metrics['overall_fid']:.4f}")
print(f"Fidelity: {metrics['label_fidelity']:.4f}")

# Per-class details
for class_name, fidelity in metrics['per_class_label_fidelity'].items():
    print(f"  {class_name}: {fidelity:.2%}")
```

## Related Scripts

- `scripts/evaluate.py` - Generate metrics.json from a trained model
- `scripts/train.py` - Train a GAN model
- `src/eval/evaluate.py` - Core evaluation framework

## Questions?

See `EVALUATION.md` for detailed metric explanations or `EVALUATION_IMPLEMENTATION.md` for technical details.
