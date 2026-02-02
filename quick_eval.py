#!/usr/bin/env python
"""
QUICK START: Run complete evaluation in 2 minutes

This script demonstrates running full evaluation with minimal parameters.
Generates 20 images per class and trains classifier for 3 epochs.
"""

import subprocess
import sys

def run_eval(model_name):
    """Run evaluation for a model"""
    
    print("\n" + "="*70)
    print(f"🚀 QUICK EVALUATION: {model_name.upper()}")
    print("="*70)
    
    config_map = {
        "cgan": {
            "config": "configs/cgan_baseline_128.yaml",
            "run_name": "cgan_baseline_128"
        },
        "qcgan": {
            "config": "configs/qcgan_baseline_128.yaml",
            "run_name": "qcgan_baseline_128"
        }
    }
    
    if model_name not in config_map:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    cfg = config_map[model_name]
    
    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--model", model_name,
        "--run_name", cfg["run_name"],
        "--config", cfg["config"],
        "--num_images_per_class", "20",  # Small for quick test
        "--classifier_epochs", "3",       # Few epochs for quick test
        "--batch_size", "32"
    ]
    
    print(f"\n✓ Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=".")
    
    return result.returncode == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick evaluation test")
    parser.add_argument("--model", type=str, choices=["cgan", "qcgan"], required=True)
    args = parser.parse_args()
    
    success = run_eval(args.model)
    
    if success:
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: runs/{args.model}_baseline_128/evaluation/")
        print("  - metrics.json")
        print("  - metrics.csv")
        print("  - generated/")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)
