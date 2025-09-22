#!/usr/bin/env python3
"""
Full I3D Violence Detection Training Script
Trains on the organized crowd_dataset with proper data loading and training loop.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    """Dataset for loading video clips for violence detection training."""
    
    def __init__(self, data_dir, split='train', clip_length=16, fps=8, resolution=224):
        self.data_dir = Path(data_dir)
        self.split = split
        self.clip_length = clip_length
        self.fps = fps
        self.resolution = resolution
        
        # Get class directories
        self.classes = sorted([d.name for d in (self.data_dir / split).iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all video files
        self.video_files = []
        for cls in self.classes:
            cls_dir = self.data_dir / split / cls
            for video_file in cls_dir.glob('*'):
                if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self.video_files.append((video_file, self.class_to_idx[cls]))
        
        logger.info(f"Found {len(self.video_files)} videos in {split} split")
        logger.info(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        
        # Extract frames from video
        frames = self._extract_frames(video_path)
        
        if frames is None or len(frames) < self.clip_length:
            # If video is too short, repeat frames
            if frames is not None and len(frames) > 0:
                frames = np.tile(frames, (self.clip_length // len(frames) + 1, 1, 1, 1))[:self.clip_length]
            else:
                # Create dummy frames if video is corrupted
                frames = np.zeros((self.clip_length, self.resolution, self.resolution, 3), dtype=np.uint8)
        
        # Convert to tensor and normalize
        frames = torch.from_numpy(frames).float() / 255.0
        # (T, H, W, C) -> (T, C, H, W) so DataLoader yields (B, T, C, H, W)
        # But model expects (B, C, T, H, W), so we'll permute in forward pass
        frames = frames.permute(0, 3, 1, 2)
        
        return frames, label
    
    def _extract_frames(self, video_path):
        """Extract frames from video at specified FPS."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_fps <= 0:
            logger.warning(f"Invalid FPS for video: {video_path}")
            cap.release()
            return None
        
        # Calculate frame interval for target FPS
        frame_interval = max(1, int(video_fps / self.fps))
        
        frames = []
        frame_idx = 0
        
        while len(frames) < self.clip_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Resize frame
                frame = cv2.resize(frame, (self.resolution, self.resolution))
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        return np.array(frames) if frames else None

class I3DModel(nn.Module):
    """Simplified I3D model for violence detection."""
    
    def __init__(self, num_classes=2, input_channels=3, clip_length=16):
        super(I3DModel, self).__init__()
        
        self.clip_length = clip_length
        self.num_classes = num_classes
        
        # 3D Convolutional layers
        self.conv3d_1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), 
                                 stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), 
                                 stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3d_3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), 
                                 stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, time, channels, height, width)
        # Convert to (batch_size, channels, time, height, width) for 3D conv
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d_1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv3d_2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3d_3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='Train I3D Violence Detection Model')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                       help='Path to training config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Coerce config types to ensure numeric values are floats/ints
    # This guards against YAML parsing edge-cases where scientific notation may be read as strings
    config.setdefault('training', {})
    training_cfg = config['training']
    # Learning rate and weight decay
    if 'learning_rate' in training_cfg:
        try:
            training_cfg['learning_rate'] = float(training_cfg['learning_rate'])
        except Exception:
            training_cfg['learning_rate'] = 1e-4
    else:
        training_cfg['learning_rate'] = 1e-4

    if 'weight_decay' in training_cfg:
        try:
            training_cfg['weight_decay'] = float(training_cfg['weight_decay'])
        except Exception:
            training_cfg['weight_decay'] = 1e-4
    else:
        training_cfg['weight_decay'] = 1e-4

    # Batch size and epochs
    if 'batch_size' in training_cfg:
        try:
            training_cfg['batch_size'] = int(training_cfg['batch_size'])
        except Exception:
            training_cfg['batch_size'] = 4
    else:
        training_cfg['batch_size'] = 4

    if 'epochs' in training_cfg:
        try:
            training_cfg['epochs'] = int(training_cfg['epochs'])
        except Exception:
            training_cfg['epochs'] = 50
    else:
        training_cfg['epochs'] = 50
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create model
    model = I3DModel(
        num_classes=2,
        input_channels=3,
        clip_length=config['input']['clip_length']
    ).to(device)
    
    # Create datasets
    train_dataset = VideoDataset(
        data_dir=config['dataset']['root'],
        split='train',
        clip_length=config['input']['clip_length'],
        fps=config['input']['fps'],
        resolution=config['input']['resolution']
    )
    
    val_dataset = VideoDataset(
        data_dir=config['dataset']['root'],
        split='val',
        clip_length=config['input']['clip_length'],
        fps=config['input']['fps'],
        resolution=config['input']['resolution']
    )
    
    # Create data loaders
    # For macOS/python-multiprocessing stability on CPU, prefer num_workers=0
    cpu_num_workers = 0 if device.type == 'cpu' else 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=cpu_num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=cpu_num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_cfg['epochs']
    )
    
    # Training loop
    best_val_acc = 0.0
    patience = training_cfg.get('early_stopping_patience', 10)
    patience_counter = 0
    
    logger.info(f"Starting training for {training_cfg['epochs']} epochs...")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    for epoch in range(1, training_cfg['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{training_cfg['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs(os.path.dirname(config['model']['checkpoint_path']), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, config['model']['checkpoint_path'])
            
            logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {config['model']['checkpoint_path']}")

if __name__ == '__main__':
    main()
