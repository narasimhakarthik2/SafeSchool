import torch
from data.dataset import create_dataloader
from training.trainer import Trainer
import yaml
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # Verify dataset directory exists
    if not os.path.exists(config['data']['processed_dir']):
        raise FileNotFoundError(
            f"Dataset directory not found at {config['data']['processed_dir']}. "
            "Please provide --raw_data argument to organize the dataset."
        )

    # Create dataloaders
    train_loader = create_dataloader(
        root_dir=config['data']['processed_dir'],
        split='train',
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        img_size=config['data']['img_size']
    )

    val_loader = create_dataloader(
        root_dir=config['data']['processed_dir'],
        split='val',
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        img_size=config['data']['img_size']
    )

    # Initialize trainer
    trainer = Trainer(config)

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()