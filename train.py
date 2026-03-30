import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss functions
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=255)
        self.criterion_dice = DiceLoss(num_classes=config['num_classes'])
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['epochs']
        )
        
        # Initialize wandb
        wandb.init(project="face-parsing", config=config)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_dice_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            # Calculate losses
            ce_loss = self.criterion_ce(outputs, masks)
            dice_loss = self.criterion_dice(outputs, masks)
            loss = ce_loss + dice_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()
            
            pbar.set_postfix({'Loss': loss.item()})
        
        return {
            'loss': total_loss / len(self.train_loader),
            'ce_loss': total_ce_loss / len(self.train_loader),
            'dice_loss': total_dice_loss / len(self.train_loader)
        }
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_dice_loss = 0
        total_iou = 0
        total_acc = 0
        
        for images, masks in tqdm(self.val_loader, desc='Validation'):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images)
            
            # Calculate losses
            ce_loss = self.criterion_ce(outputs, masks)
            dice_loss = self.criterion_dice(outputs, masks)
            loss = ce_loss + dice_loss
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()
            
            # Calculate metrics
            preds = outputs.argmax(dim=1)
            iou = compute_iou(preds, masks, num_classes=self.config['num_classes'])
            acc = (preds == masks).float().mean().item()
            
            total_iou += iou
            total_acc += acc
        
        return {
            'loss': total_loss / len(self.val_loader),
            'ce_loss': total_ce_loss / len(self.val_loader),
            'dice_loss': total_dice_loss / len(self.val_loader),
            'iou': total_iou / len(self.val_loader),
            'acc': total_acc / len(self.val_loader)
        }
    
    def train(self):
        best_iou = 0
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            wandb.log({
                'train/loss': train_metrics['loss'],
                'train/ce_loss': train_metrics['ce_loss'],
                'train/dice_loss': train_metrics['dice_loss'],
                'val/loss': val_metrics['loss'],
                'val/iou': val_metrics['iou'],
                'val/acc': val_metrics['acc'],
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['iou']:.4f}, Val Acc: {val_metrics['acc']:.4f}")
            
            # Save best model
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_iou': best_iou,
                    'config': self.config
                }, 'best_model.pth')
                print(f"Saved best model with IoU: {best_iou:.4f}")

# Utility functions
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        # one-hot
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()

def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.mean(ious) if ious else 0