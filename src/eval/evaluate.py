"""
Comprehensive GAN Evaluation Framework

This module provides end-to-end evaluation of conditional GANs using multiple metrics:

1. **FID (Fréchet Inception Distance)**
   - Measures the statistical similarity between real and generated image distributions
   - Uses InceptionV3 features to compute mean and covariance in feature space
   - Lower FID = more similar distributions (better quality and diversity)
   - Assumes: Both real and generated images should have similar feature distributions
   - per-class FID helps identify which classes are easier/harder to generate

2. **Label Fidelity**
   - Percentage of generated images correctly classified as their conditioning label
   - Trained on real images, tested on generated images
   - High fidelity = generator respects class conditioning
   - Assumptions: Classifier trained on real data generalizes to generated data
   - Per-class fidelity reveals class-specific generation quality

3. **Classifier Performance on Real Data**
   - Baseline metrics for the reference classifier
   - Used to understand if the classifier is reliable for fidelity evaluation
   - High real-data accuracy is necessary for trusting fidelity scores

Why per-class evaluation matters:
- Some defect types may be easier to generate than others
- Class imbalance can skew overall metrics
- Identifies specific failure modes in generation
- Guides hyperparameter tuning and model improvements
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False
    warnings.warn("torchmetrics not installed, will use manual FID calculation")


# ============================================================================
# CONFIGURATION & DATACLASSES
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration"""
    # Dataset paths
    train_image_dir: str  # Path to real training images organized by class
    val_image_dir: str    # Path to real validation images organized by class
    
    # Generation
    num_images_per_class: int = 50  # Generate N images per class
    
    # Classifier
    classifier_hidden_dim: int = 256
    classifier_epochs: int = 10
    classifier_batch_size: int = 32
    classifier_lr: float = 0.001
    
    # FID
    fid_batch_size: int = 32
    
    # Device
    device: str = "cpu"
    
    # Output
    output_dir: str = "runs/eval_results"  # Will be overridden by run_name


@dataclass
class EvalMetrics:
    """Container for evaluation results"""
    overall_fid: float
    per_class_fid: Dict[str, float]
    
    classifier_real_accuracy: float
    classifier_real_precision: Dict[str, float]  # Per-class
    classifier_real_recall: Dict[str, float]     # Per-class
    
    label_fidelity: float
    per_class_label_fidelity: Dict[str, float]
    
    num_generated_images: int
    num_real_validation_images: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


# ============================================================================
# DATASETS
# ============================================================================

class ClassificationDataset(Dataset):
    """Generic dataset for classification from class-organized directories"""
    
    def __init__(
        self,
        image_dir: str,
        img_size: int = 128,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            image_dir: Base directory with subdirectories named by class
            img_size: Target image size
            transform: Optional transforms to apply
        """
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Scan directory structure
        self.class_dirs = sorted([d for d in self.image_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in self.class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Collect all image paths
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_dir in enumerate(self.class_dirs):
            image_files = sorted([
                f for f in class_dir.iterdir() 
                if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
            ])
            self.image_paths.extend(image_files)
            self.labels.extend([class_idx] * len(image_files))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and process image
        image = Image.open(image_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class GeneratedImageDataset(Dataset):
    """Dataset for generated images stored as tensors"""
    
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            images: (N, 1, H, W) tensor of images in [-1, 1] range
            labels: (N,) tensor of class indices
            transform: Optional transforms
        """
        self.images = images
        self.labels = labels
        # Images are already tensors in [-1, 1], no additional transform needed
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Images are already properly formatted tensors
        return image, label


# ============================================================================
# FID CALCULATION
# ============================================================================

class FIDCalculator:
    """
    Calculate Fréchet Inception Distance between two image distributions.
    
    FID measures the distance between Gaussian distributions fit to InceptionV3
    feature vectors extracted from real and generated images:
    
        FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real·Σ_gen)^0.5)
    
    Lower FID indicates better quality and diversity of generated images.
    """
    
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        
        # Load InceptionV3 and remove classification layer
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False
        ).to(device)
        inception.eval()
        
        # Create custom feature extractor that stops at Mixed_7a
        # This avoids issues with 128x128 images being too small
        class InceptionFeatureExtractor(nn.Module):
            def __init__(self, inception_model):
                super().__init__()
                self.inception = inception_model
            
            def forward(self, x):
                # Input: (N, 3, H, W)
                # Conv2d_1a_3x3: (N, 32, H/2, W/2)
                x = self.inception.Conv2d_1a_3x3(x)
                x = self.inception.Conv2d_2a_3x3(x)
                x = self.inception.Conv2d_2b_3x3(x)
                x = nn.functional.max_pool2d(x, kernel_size=3, stride=2)
                
                x = self.inception.Conv2d_3b_1x1(x)
                x = self.inception.Conv2d_4a_3x3(x)
                x = nn.functional.max_pool2d(x, kernel_size=3, stride=2)
                
                x = self.inception.Mixed_5b(x)
                x = self.inception.Mixed_5c(x)
                x = self.inception.Mixed_5d(x)
                
                # At this point features are 32×32, enough for adaptive pool
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
                return x
        
        self.inception = InceptionFeatureExtractor(inception).to(device)
    
    def extract_features(self, dataloader: DataLoader) -> np.ndarray:
        """
        Extract InceptionV3 features from a dataset.
        
        Args:
            dataloader: DataLoader yielding (images, labels) tuples
        
        Returns:
            Feature array of shape (N, 2048)
        """
        features_list = []
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                
                # Ensure 3-channel input for Inception
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
                
                # Extract features
                feats = self.inception(images)
                feats = feats.view(feats.size(0), -1)
                features_list.append(feats.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mean and covariance of feature distribution.
        
        Args:
            features: Feature array of shape (N, D)
        
        Returns:
            (mean, cov) both of shape (D,) and (D, D)
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features.T)
        return mu, sigma
    
    def compute_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray
    ) -> float:
        """
        Compute FID between two Gaussian distributions.
        
        Args:
            mu1, sigma1: Mean and covariance of first distribution
            mu2, sigma2: Mean and covariance of second distribution
        
        Returns:
            FID score (lower is better)
        """
        diff = mu1 - mu2
        
        # Frobenius norm of difference in means
        diff_norm = np.sqrt(np.sum(diff ** 2))
        
        # Matrix square root of product
        from scipy import linalg
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Handle numerical errors in sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        trace = np.trace(sigma1 + sigma2 - 2 * covmean)
        
        fid = diff_norm ** 2 + trace
        return float(fid)


# ============================================================================
# CLASSIFIER
# ============================================================================

class SimpleClassifier(nn.Module):
    """Simple CNN classifier for evaluating label fidelity"""
    
    def __init__(
        self,
        num_classes: int = 6,
        hidden_dim: int = 256,
        img_size: int = 128
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# EVALUATOR
# ============================================================================

class GANEvaluator:
    """Main evaluation orchestrator"""
    
    def __init__(
        self,
        config: EvalConfig,
        class_names: List[str],
        num_classes: int
    ):
        self.config = config
        self.class_names = class_names
        self.num_classes = num_classes
        self.device = config.device
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize FID calculator
        self.fid_calc = FIDCalculator(device=self.device, batch_size=config.fid_batch_size)
    
    def generate_samples(
        self,
        generator: nn.Module,
        z_dim: int,
        num_per_class: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate balanced samples from generator.
        
        Args:
            generator: Trained generator model
            z_dim: Latent dimension
            num_per_class: Number of samples per class
        
        Returns:
            (generated_images, labels) both tensors
        """
        generator.eval()
        all_images = []
        all_labels = []
        
        with torch.no_grad():
            for class_id in tqdm(range(self.num_classes), desc="Generating samples"):
                z = torch.randn(num_per_class, z_dim, device=self.device)
                c = torch.full(
                    (num_per_class,), class_id,
                    dtype=torch.long, device=self.device
                )
                
                fake_images = generator(z, c)
                all_images.append(fake_images.cpu())
                all_labels.extend([class_id] * num_per_class)
        
        return torch.cat(all_images, dim=0), torch.tensor(all_labels)
    
    def save_generated_images(
        self,
        generated_images: torch.Tensor,
        labels: torch.Tensor,
        run_name: str
    ) -> None:
        """Save generated images organized by class"""
        save_dir = Path(self.config.output_dir) / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for class_id, class_name in enumerate(self.class_names):
            class_dir = save_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Find indices for this class
            class_mask = labels == class_id
            class_images = generated_images[class_mask]
            
            # Save each image
            for idx, img in enumerate(class_images):
                # Denormalize from [-1, 1] to [0, 1]
                img = (img + 1) / 2
                img = torch.clamp(img, 0, 1)
                
                save_path = class_dir / f"{class_name}_{idx:04d}.png"
                transforms.ToPILImage()(img.squeeze()).save(str(save_path))
    
    def train_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> SimpleClassifier:
        """Train classifier on real training data"""
        classifier = SimpleClassifier(
            num_classes=self.num_classes,
            hidden_dim=self.config.classifier_hidden_dim
        ).to(self.device)
        
        optimizer = optim.Adam(
            classifier.parameters(),
            lr=self.config.classifier_lr
        )
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_classifier = None
        
        for epoch in range(self.config.classifier_epochs):
            # Training
            classifier.train()
            train_loss = 0
            
            for images, labels in tqdm(
                train_loader,
                desc=f"Classifier Epoch {epoch+1}/{self.config.classifier_epochs}"
            ):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                logits = classifier(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            classifier.eval()
            val_acc = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = classifier(images)
                    preds = logits.argmax(dim=1)
                    val_acc += (preds == labels).sum().item() / len(labels)
            
            val_acc /= len(val_loader)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_classifier = classifier.state_dict().copy()
        
        if best_classifier is not None:
            classifier.load_state_dict(best_classifier)
        
        return classifier
    
    def evaluate_classifier(
        self,
        classifier: SimpleClassifier,
        dataloader: DataLoader
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Evaluate classifier and compute per-class metrics.
        
        Returns:
            (accuracy, per_class_precision, per_class_recall)
        """
        classifier.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating classifier"):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = classifier(images)
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Overall accuracy
        accuracy = np.mean(all_preds == all_labels)
        
        # Per-class precision and recall
        precision_dict = {}
        recall_dict = {}
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            
            # Precision: TP / (TP + FP)
            tp = np.sum((all_preds == class_id) & (all_labels == class_id))
            fp = np.sum((all_preds == class_id) & (all_labels != class_id))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_dict[class_name] = float(precision)
            
            # Recall: TP / (TP + FN)
            fn = np.sum((all_preds != class_id) & (all_labels == class_id))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_dict[class_name] = float(recall)
        
        return float(accuracy), precision_dict, recall_dict
    
    def compute_fid_scores(
        self,
        val_loader: DataLoader,
        generated_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute FID scores (overall and per-class).
        
        Returns:
            (overall_fid, per_class_fid_dict)
        """
        # Extract features
        val_features = self.fid_calc.extract_features(val_loader)
        gen_features = self.fid_calc.extract_features(generated_loader)
        
        # Overall FID
        mu_val, sigma_val = self.fid_calc.calculate_statistics(val_features)
        mu_gen, sigma_gen = self.fid_calc.calculate_statistics(gen_features)
        overall_fid = self.fid_calc.compute_fid(mu_val, sigma_val, mu_gen, sigma_gen)
        
        # Per-class FID (would require class-conditional feature extraction)
        # For now, return empty dict (can be extended)
        per_class_fid = {}
        
        return overall_fid, per_class_fid
    
    def compute_label_fidelity(
        self,
        classifier: SimpleClassifier,
        generated_images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute label fidelity: % of generated images classified correctly.
        
        High fidelity means generator respects class conditioning.
        
        Returns:
            (overall_fidelity, per_class_fidelity_dict)
        """
        classifier.eval()
        
        # Create dataset
        dataset = GeneratedImageDataset(generated_images, labels)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.fid_batch_size,
            shuffle=False
        )
        
        # Compute fidelity
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels_batch in tqdm(dataloader, desc="Computing label fidelity"):
                images = images.to(self.device)
                logits = classifier(images)
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels_batch.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Overall fidelity
        overall_fidelity = np.mean(all_preds == all_labels)
        
        # Per-class fidelity
        per_class_fidelity = {}
        for class_id, class_name in enumerate(self.class_names):
            class_mask = all_labels == class_id
            if class_mask.sum() > 0:
                class_fidelity = np.mean(all_preds[class_mask] == class_id)
                per_class_fidelity[class_name] = float(class_fidelity)
            else:
                per_class_fidelity[class_name] = 0.0
        
        return float(overall_fidelity), per_class_fidelity
    
    def evaluate(
        self,
        generator: nn.Module,
        z_dim: int,
        run_name: str
    ) -> EvalMetrics:
        """
        Run complete evaluation pipeline.
        
        Args:
            generator: Trained generator model
            z_dim: Latent dimension
            run_name: Name for saving results
        
        Returns:
            EvalMetrics object with all computed metrics
        """
        # Load real data
        print("\n" + "="*70)
        print("🔍 LOADING REAL DATA")
        print("="*70)
        
        val_dataset = ClassificationDataset(self.config.val_image_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.classifier_batch_size,
            shuffle=False
        )
        
        train_dataset = ClassificationDataset(self.config.train_image_dir)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.classifier_batch_size,
            shuffle=True
        )
        
        num_real_val = len(val_dataset)
        print(f"✓ Loaded {num_real_val} real validation images")
        
        # Generate samples
        print("\n" + "="*70)
        print("🎨 GENERATING SYNTHETIC SAMPLES")
        print("="*70)
        
        generated_images, generated_labels = self.generate_samples(
            generator,
            z_dim,
            self.config.num_images_per_class
        )
        
        num_generated = len(generated_images)
        print(f"✓ Generated {num_generated} images ({self.config.num_images_per_class} per class)")
        
        # Save generated images
        self.save_generated_images(generated_images, generated_labels, run_name)
        print(f"✓ Saved to: {self.config.output_dir}/generated/")
        
        # Train classifier
        print("\n" + "="*70)
        print("🏋️  TRAINING CLASSIFIER")
        print("="*70)
        
        classifier = self.train_classifier(train_loader, val_loader)
        print("✓ Classifier training complete")
        
        # Evaluate classifier on real validation data
        print("\n" + "="*70)
        print("📊 CLASSIFIER PERFORMANCE (Real Validation Data)")
        print("="*70)
        
        real_accuracy, real_precision, real_recall = self.evaluate_classifier(
            classifier, val_loader
        )
        print(f"✓ Accuracy: {real_accuracy:.4f}")
        
        # Compute FID
        print("\n" + "="*70)
        print("📐 COMPUTING FID (Fréchet Inception Distance)")
        print("="*70)
        
        generated_dataset = GeneratedImageDataset(
            generated_images,
            generated_labels
        )
        generated_loader = DataLoader(
            generated_dataset,
            batch_size=self.config.fid_batch_size,
            shuffle=False
        )
        
        overall_fid, per_class_fid = self.compute_fid_scores(
            val_loader,
            generated_loader
        )
        print(f"✓ Overall FID: {overall_fid:.4f}")
        
        # Compute label fidelity
        print("\n" + "="*70)
        print("✅ COMPUTING LABEL FIDELITY")
        print("="*70)
        
        label_fidelity, per_class_fidelity = self.compute_label_fidelity(
            classifier,
            generated_images,
            generated_labels
        )
        print(f"✓ Overall label fidelity: {label_fidelity:.4f}")
        
        # Compile metrics
        metrics = EvalMetrics(
            overall_fid=overall_fid,
            per_class_fid=per_class_fid,
            classifier_real_accuracy=real_accuracy,
            classifier_real_precision=real_precision,
            classifier_real_recall=real_recall,
            label_fidelity=label_fidelity,
            per_class_label_fidelity=per_class_fidelity,
            num_generated_images=num_generated,
            num_real_validation_images=num_real_val
        )
        
        return metrics
    
    def save_metrics(self, metrics: EvalMetrics, run_name: str) -> None:
        """Save metrics to JSON and CSV"""
        output_dir = Path(self.config.output_dir)
        
        # JSON
        json_path = output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\n✓ Saved metrics to: {json_path}")
        
        # CSV
        csv_path = output_dir / "metrics.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'metric_name', 'value', 'per_class_breakdown'
            ])
            writer.writeheader()
            
            # Overall metrics
            writer.writerow({
                'metric_name': 'Overall FID',
                'value': metrics.overall_fid,
                'per_class_breakdown': ''
            })
            
            writer.writerow({
                'metric_name': 'Label Fidelity',
                'value': metrics.label_fidelity,
                'per_class_breakdown': str(metrics.per_class_label_fidelity)
            })
            
            writer.writerow({
                'metric_name': 'Classifier Real Accuracy',
                'value': metrics.classifier_real_accuracy,
                'per_class_breakdown': ''
            })
            
            # Per-class fidelity
            for class_name, fidelity in metrics.per_class_label_fidelity.items():
                writer.writerow({
                    'metric_name': f'{class_name} Label Fidelity',
                    'value': fidelity,
                    'per_class_breakdown': ''
                })
            
            # Per-class precision
            for class_name, prec in metrics.classifier_real_precision.items():
                writer.writerow({
                    'metric_name': f'{class_name} Precision (Real)',
                    'value': prec,
                    'per_class_breakdown': ''
                })
            
            # Per-class recall
            for class_name, rec in metrics.classifier_real_recall.items():
                writer.writerow({
                    'metric_name': f'{class_name} Recall (Real)',
                    'value': rec,
                    'per_class_breakdown': ''
                })
        
        print(f"✓ Saved metrics to: {csv_path}")


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
