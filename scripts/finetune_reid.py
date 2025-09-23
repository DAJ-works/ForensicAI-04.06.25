import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.reid_trainer import ReIDTrainer, ReIDDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune person re-identification model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training dataset directory")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/reid", help="Output directory for checkpoints")
    parser.add_argument("--model-type", type=str, choices=['osnet', 'mgn'], default='osnet', help="Model architecture")
    parser.add_argument("--feature-dim", type=int, default=512, help="Feature dimension")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, choices=['adam', 'sgd'], default='adam', help="Optimizer")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model weights")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--attribute-file", type=str, default=None, help="Path to attribute annotations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save-freq", type=int, default=10, help="Save checkpoint frequency (epochs)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID (-1 for CPU)")
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up dataset paths
    dataset_dir = Path(args.dataset)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    
    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        return 1
    
    # Determine if using attributes
    use_attributes = args.attribute_file is not None
    
    # Count number of identities from training directory
    num_classes = len(list(train_dir.glob("*")))
    logger.info(f"Found {num_classes} identity classes in training set")
    
    if num_classes == 0:
        logger.error("No identity classes found")
        return 1
    
    # Create data loaders
    train_dataset = ReIDDataset(
        data_dir=str(train_dir),
        is_train=True,
        use_attributes=use_attributes,
        attribute_file=args.attribute_file
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create validation loader if test directory exists
    val_loader = None
    if test_dir.exists():
        test_dataset = ReIDDataset(
            data_dir=str(test_dir),
            is_train=False,
            use_attributes=use_attributes,
            attribute_file=args.attribute_file
        )
        
        val_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Validation set: {len(test_dataset)} images")
    
    # Get attribute dimensions if using attributes
    attribute_dims = None
    if use_attributes:
        try:
            with open(args.attribute_file, 'r') as f:
                attributes = json.load(f)
            
            # Get a sample to determine attribute dimensions
            sample_key = next(iter(attributes))
            sample_attrs = attributes[sample_key]
            
            # Create attribute dimensions dictionary
            attribute_dims = {}
            for attr_name, attr_val in sample_attrs.items():
                if isinstance(attr_val, int):
                    # Count unique values in the dataset
                    unique_vals = set(attrs.get(attr_name, 0) for attrs in attributes.values() 
                                    if attr_name in attrs)
                    attribute_dims[attr_name] = max(2, len(unique_vals))
                else:
                    # For more complex attributes, use default
                    attribute_dims[attr_name] = 2
            
            logger.info(f"Attribute dimensions: {attribute_dims}")
        except Exception as e:
            logger.error(f"Failed to process attribute file: {e}")
            use_attributes = False
    
    # Create trainer
    trainer = ReIDTrainer(
        model_type=args.model_type,
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        pretrained=args.pretrained,
        use_attributes=use_attributes,
        attribute_dims=attribute_dims,
        device=device
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        try:
            trainer.load_model(args.checkpoint)
            logger.info(f"Resumed from checkpoint: {args.checkpoint}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    # Save training configuration
    config = {
        'dataset': str(dataset_dir),
        'model_type': args.model_type,
        'feature_dim': args.feature_dim,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'pretrained': args.pretrained,
        'use_attributes': use_attributes,
        'attribute_dims': attribute_dims,
        'datetime': datetime.now().isoformat(),
        'num_classes': num_classes,
        'num_train_images': len(train_dataset),
        'num_val_images': len(test_dataset) if val_loader else 0
    }
    
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    logger.info("Starting training")
    start_time = time.time()
    
    stats = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        save_dir=str(output_dir),
        save_freq=args.save_freq
    )
    
    # Training complete
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    if val_loader and 'best_accuracy' in stats:
        logger.info(f"Best validation accuracy: {stats['best_accuracy']:.2f}%")
    
    # Save final result summary
    with open(output_dir / "training_summary.json", 'w') as f:
        json.dump({
            'config': config,
            'stats': stats,
            'training_time': total_time,
            'training_time_formatted': f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        }, f, indent=2)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())