#!/usr/bin/env python3
"""
Monitor CGAN Training Progress

Run this while training to see current status:
    python3 monitor_training.py
"""

import time
import csv
from pathlib import Path
import os

LOG_FILE = Path("runs/cgan_baseline_128/logs/train_log.csv")
SAMPLES_DIR = Path("runs/cgan_baseline_128/samples")

def monitor_training():
    print("\n" + "="*70)
    print("CGAN TRAINING MONITOR")
    print("="*70 + "\n")
    
    while True:
        try:
            if LOG_FILE.exists():
                # Read latest training log
                with open(LOG_FILE, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        # Get last line
                        last_line = lines[-1].strip()
                        epoch, d_loss, g_loss = last_line.split(',')
                        
                        # Count sample images generated
                        if SAMPLES_DIR.exists():
                            num_samples = len(list(SAMPLES_DIR.glob("*.png")))
                        else:
                            num_samples = 0
                        
                        print(f"\r📊 TRAINING STATUS:")
                        print(f"   Epoch: {epoch}/500")
                        print(f"   D Loss: {float(d_loss):.6f}")
                        print(f"   G Loss: {float(g_loss):.6f}")
                        print(f"   Samples Generated: {num_samples}")
                        print("\n   (Press Ctrl+C to exit)")
                        
                        time.sleep(5)  # Update every 5 seconds
            else:
                print("⏳ Waiting for training to start...")
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n✓ Monitor stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    monitor_training()
