"""
ARGUS Training Script
Optimized for phish_sample_30k dataset structure
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import re
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from PIL import Image
import random
from typing import Optional, Tuple

# Import core modules
from argus.utils.config import Config
from argus.utils.runtime import seed_everything
from argus.core.feature_extractor import PhishingFeatureExtractor
from argus.core.prior_trigger import PriorTriggerRules
from argus.core.learnable_trigger import LearnableTrigger
from argus.core.confidence_estimator import ConfidenceEstimator
from argus.core.multimodal_fusion import OmniModalFusion
from argus.training.trainer import ARGUSTrainer
from argus.training.adversarial_aug import AdversarialAugmentation
from argus.training.data_validation import collate_samples, count_binary_labels, validate_label_counts


class PhishingSampleDataset(Dataset):
    """
    Dataset loader for phish_sample_30k structure

    Each sample folder contains:
        - html.txt: HTML content
        - info.txt: URL and metadata
        - shot.png: Screenshot
        - ocr.txt: OCR text (optional)
        - yolo_coords.txt: YOLO coordinates (optional)
    """

    def __init__(self,
                 dataset_root: str,
                 feature_extractor: PhishingFeatureExtractor,
                 prior_trigger: PriorTriggerRules,
                 split: str = 'train',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 augment: bool = False,
                 max_samples: Optional[int] = None,
                 seed: int = 42,
                 default_label: Optional[int] = None):
        """
        Args:
            dataset_root: Path to phish_sample_30k folder
            feature_extractor: Feature extractor instance
            prior_trigger: Prior trigger instance
            split: 'train', 'val', or 'test'
            train_ratio: Training set ratio (default: 0.7)
            val_ratio: Validation set ratio (default: 0.15)
            augment: Whether to use data augmentation
            max_samples: Maximum samples to load (None = all)
            seed: Random seed for reproducibility
            default_label: Explicit 0/1 policy for missing labels (default: reject)
        """
        self.dataset_root = Path(dataset_root)
        self.feature_extractor = feature_extractor
        self.prior_trigger = prior_trigger
        self.split = split
        self.augment = augment
        self.seed = seed
        if default_label not in (None, 0, 1):
            raise ValueError('default_label must be explicitly 0, 1, or None')
        self.default_label = default_label

        if augment:
            self.augmentor = AdversarialAugmentation()

        # Load and split samples
        self.samples = self._load_and_split_samples(
            train_ratio, val_ratio, max_samples
        )

        metadata = [self._read_metadata(folder) for folder in self.samples]
        self.labels = [item[1] for item in metadata]
        self.label_counts = count_binary_labels(self.labels)
        self.defaulted_label_count = sum(item[2] for item in metadata)
        print(f"{split}: label_counts={self.label_counts}, explicit_default_labels={self.defaulted_label_count}")
        validate_label_counts(self.label_counts, split, require_both=split != 'train')
        print(f"✅ Loaded {len(self.samples)} samples for '{split}' split")

    def _load_and_split_samples(self,
                                train_ratio: float,
                                val_ratio: float,
                                max_samples: Optional[int]) -> list:
        """Load sample folders and split into train/val/test"""

        # Get all valid sample folders
        all_folders = []

        if not self.dataset_root.exists():
            raise ValueError(f"Dataset root not found: {self.dataset_root}")

        for folder in self.dataset_root.iterdir():
            if not folder.is_dir():
                continue

            # Check for required files
            info_file = folder / 'info.txt'
            html_file = folder / 'html.txt'

            # At least one of these should exist
            if info_file.exists() or html_file.exists():
                all_folders.append(folder)

        if len(all_folders) == 0:
            raise ValueError(f"No valid samples found in {self.dataset_root}")

        print(f"📊 Found {len(all_folders)} total samples in dataset")

        # Shuffle with fixed seed for reproducibility
        random.seed(self.seed)
        random.shuffle(all_folders)

        # Limit samples if specified
        if max_samples is not None and max_samples < len(all_folders):
            all_folders = all_folders[:max_samples]
            print(f"📊 Using {max_samples} samples (limited by max_samples)")

        # Calculate split sizes
        n_total = len(all_folders)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # n_test = n_total - n_train - n_val (remaining)

        # Split based on requested split
        if self.split == 'train':
            selected = all_folders[:n_train]
        elif self.split == 'val':
            selected = all_folders[n_train:n_train + n_val]
        elif self.split == 'test':
            selected = all_folders[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")

        return selected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Load and process a single sample"""
        sample_folder = self.samples[idx]

        # Load data from folder
        try:
            url, html, image, label = self._load_sample_files(sample_folder)
        except Exception as error:
            return self._failure(sample_folder, 'files', error)

        # Extract 100-dimensional features
        try:
            features = self.feature_extractor.extract_features(url, html)
            features = torch.from_numpy(features).float()
            if features.shape != (100,) or not torch.isfinite(features).all():
                raise ValueError('Expected finite features with shape (100,)')
        except Exception as e:
            return self._failure(sample_folder, 'feature', e)

        # Compute prior trigger scores
        try:
            prior_scores = self.prior_trigger.compute_trigger_scores(
                features.unsqueeze(0)
            ).squeeze(0)
            if prior_scores.shape != (len(self.prior_trigger.task_names),) or not torch.isfinite(prior_scores).all():
                raise ValueError('Expected finite prior scores with one entry per task')
        except Exception as e:
            return self._failure(sample_folder, 'prior', e)

        # Build return dictionary
        item = {
            'features': features,
            'prior_scores': prior_scores,
            'label': torch.tensor(label, dtype=torch.long),
            'url': url,
            'folder_name': sample_folder.name
        }

        # Add augmented image if available
        if image is not None and self.augment:
            try:
                augmented_image, is_manipulated = self.augmentor(image, p=0.5)
                item['is_manipulated'] = is_manipulated
                # Note: Could add image tensor here for visual models
            except Exception as e:
                print(f"Warning: Image augmentation failed for {sample_folder.name}: {e}")

        return item

    @staticmethod
    def _failure(folder, stage, error):
        return {'folder_name': folder.name, 'error_stage': stage,
                'error': f'{type(error).__name__}: {error}'}

    def _read_metadata(self, folder):
        """Accept explicit label fields/standalone tokens, never URL substrings."""
        info_file = folder / 'info.txt'
        content = info_file.read_text(encoding='utf-8') if info_file.exists() else ''
        url = None
        labels = []
        label_map = {'0': 0, 'benign': 0, 'legitimate': 0, 'legit': 0,
                     '1': 1, 'phishing': 1, 'phish': 1, 'malicious': 1}
        for line in content.splitlines():
            line = line.strip()
            if line.lower().startswith('url:'):
                url = line.split(':', 1)[1].strip()
                continue
            if line.startswith(('http://', 'https://')):
                url = line
                continue
            match = re.fullmatch(r'(?:label|class)\s*:\s*(.*)', line, re.IGNORECASE)
            token = match.group(1).strip().lower() if match else line.lower()
            if match or token in label_map:
                if token not in label_map:
                    raise ValueError(f'{folder.name}: unrecognized explicit label {token!r}')
                labels.append(label_map[token])
        if len(set(labels)) > 1:
            raise ValueError(f'{folder.name}: conflicting labels')
        defaulted = not labels
        if defaulted and self.default_label is None:
            raise ValueError(f'{folder.name}: missing explicit label; supply label: 0/1 or an explicit default_label policy')
        if not url or not url.startswith(('http://', 'https://')) or not urlparse(url).hostname:
            raise ValueError(f'{folder.name}: missing valid URL; synthetic URLs are not permitted')
        return url, labels[0] if labels else self.default_label, defaulted

    def _load_sample_files(self, folder: Path) -> Tuple[str, Optional[str], Optional[Image.Image], int]:
        """Load observed metadata/content; failures are returned by __getitem__."""
        url, label, _ = self._read_metadata(folder)
        html = None
        image = None

        # 2. Load html.txt
        html_file = folder / 'html.txt'
        if html_file.exists():
            try:
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    html = f.read()
                    # Skip if HTML is empty or too short
                    if len(html.strip()) < 10:
                        html = None
            except Exception as e:
                raise ValueError(f"Could not read {html_file}: {e}") from e

        # 3. Load shot.png
        shot_file = folder / 'shot.png'
        if shot_file.exists():
            try:
                image = Image.open(shot_file).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load {shot_file}: {e}")
                image = None

        return url, html, image, label


def create_dataloaders(config: Config,
                      feature_extractor: PhishingFeatureExtractor,
                      prior_trigger: PriorTriggerRules,
                      args) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""

    print(f"\n📂 Creating data loaders...")
    print(f"   Train ratio: {args.train_ratio}")
    print(f"   Val ratio: {args.val_ratio}")
    print(f"   Test ratio: {1 - args.train_ratio - args.val_ratio:.2f}")

    # Training dataset
    train_dataset = PhishingSampleDataset(
        dataset_root=args.data_dir,
        feature_extractor=feature_extractor,
        prior_trigger=prior_trigger,
        split='train',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        augment=True,
        max_samples=args.max_samples,
        seed=args.seed,
        default_label=getattr(args, 'default_label', None)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=config.device.type == 'cuda',
        drop_last=False,
        collate_fn=collate_samples
    )

    # Validation dataset
    val_dataset = PhishingSampleDataset(
        dataset_root=args.data_dir,
        feature_extractor=feature_extractor,
        prior_trigger=prior_trigger,
        split='val',
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        augment=False,
        max_samples=args.max_samples,
        seed=args.seed,
        default_label=getattr(args, 'default_label', None)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=config.device.type == 'cuda',
        drop_last=False,
        collate_fn=collate_samples
    )

    return train_loader, val_loader


def main(args):
    """Main training pipeline"""

    print("=" * 80)
    print("ARGUS Training Pipeline - Phishing Sample Dataset")
    print("=" * 80)

    # Configuration
    config = Config()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr

    print(f"\n⚙️  Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Dataset: {args.data_dir}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Top-k tasks: {config.top_k_tasks}")
    print(f"  Max samples: {args.max_samples or 'All'}")
    print(f"  Random seed: {args.seed}")

    # Check dataset exists
    dataset_path = Path(args.data_dir)
    if not dataset_path.exists():
        print(f"\n❌ Error: Dataset not found at {args.data_dir}")
        raise ValueError(f"Dataset not found: {args.data_dir}")

    # Initialize components
    print(f"\n🔧 Initializing components...")

    feature_extractor = PhishingFeatureExtractor()
    prior_trigger = PriorTriggerRules(task_names=config.TASK_NAMES)

    learnable_trigger = LearnableTrigger(
        feature_dim=config.num_features,
        num_tasks=config.num_tasks,
        prior_weight_init=config.prior_weight_init
    ).to(device=config.device, dtype=torch.float32)

    confidence_estimator = ConfidenceEstimator().to(device=config.device, dtype=torch.float32)
    multimodal_fusion = OmniModalFusion().to(device=config.device, dtype=torch.float32)

    print(f"✅ All components initialized")

    # Create data loaders
    try:
        train_loader, val_loader = create_dataloaders(
            config, feature_extractor, prior_trigger, args
        )
    except Exception as e:
        print(f"\n❌ Error creating data loaders: {e}")
        raise

    print(f"\n✅ Data loaded successfully:")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    # Initialize trainer
    trainer = ARGUSTrainer(
        learnable_trigger=learnable_trigger,
        confidence_estimator=confidence_estimator,
        multimodal_fusion=multimodal_fusion,
        config=config
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        try:
            start_epoch = trainer.load_checkpoint(args.resume) + 1
            print(f"✅ Resumed from epoch {start_epoch-1}")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print(f"   Starting from scratch...")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print(f"\n💾 Checkpoints will be saved to: {run_dir}")

    # Training loop
    print(f"\n🚀 Starting training from epoch {start_epoch}...")
    print("=" * 80)

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, config.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*80}")

        # Training
        train_metrics = trainer.train_epoch(train_loader, epoch)

        print(f"\n📊 Training Metrics:")
        print(f"  Average Loss: {train_metrics['avg_loss']:.4f}")
        print(f"  Detection Loss: {train_metrics['avg_detection_loss']:.4f}")
        print(f"  Learning Rate: {train_metrics['learning_rate']:.6f}")

        # Evaluation
        if epoch % args.eval_interval == 0:
            print(f"\n🔍 Evaluating on validation set...")
            val_metrics = trainer.evaluate(val_loader)

            print(f"\n📈 Validation Metrics:")
            print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall:    {val_metrics['recall']:.4f}")
            print(f"  F1 Score:  {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_epoch = epoch

                best_path = run_dir / 'best_model.pth'
                trainer.save_checkpoint(best_path, epoch, val_metrics)

                print(f"\n🎉 New best model! F1: {best_f1:.4f}")

        # Periodic checkpoint save
        if epoch % args.save_interval == 0:
            checkpoint_path = run_dir / f'checkpoint_epoch_{epoch}.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, train_metrics)
            print(f"💾 Checkpoint saved: epoch_{epoch}.pth")

    # Training completed
    print("\n" + "=" * 80)
    print("✅ Training Completed!")
    print("=" * 80)
    print(f"  Best F1 Score: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"  Models saved in: {run_dir}")

    # Print interpretability information
    print(f"\n📊 Model Interpretability:")
    print("-" * 80)

    interp_info = learnable_trigger.get_interpretability_info()
    print(f"  Trigger - Prior Weight: {interp_info['prior_weight']:.3f}")
    print(f"  Trigger - Learned Weight: {interp_info['learned_weight']:.3f}")

    fusion_weights = multimodal_fusion.get_base_weights()
    print(f"\n  Fusion Weights:")
    for modality, weight in fusion_weights.items():
        print(f"    {modality}: {weight:.3f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ARGUS on phish_sample_30k dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset parameters
    parser.add_argument('--data-dir', type=str,
                        default='datasets/phish_sample_30k',
                        help='Path to phish_sample_30k dataset directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of data for training (0.0-1.0)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Ratio of data for validation (0.0-1.0)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use (None for all)')

    parser.add_argument('--default-label', type=int, choices=[0, 1], default=None,
                        help='Explicit label policy for samples without labels; never inferred by default')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and validation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='Evaluate on validation set every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Host and selected-backend RNGs share one explicit runtime policy.
    seed_everything(args.seed, Config.device)

    # Start training
    main(args)