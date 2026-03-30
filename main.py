import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Prepare dataset split
        from dataset_setup import prepare_dataset_structure
        train_path, val_path, test_path = prepare_dataset_structure(args.data_root)
        
        # Create datasets
        from dataset import FaceParsingDataset, get_transforms
        train_dataset = FaceParsingDataset(
            args.data_root, 'train', transform=get_transforms('train')
        )
        val_dataset = FaceParsingDataset(
            args.data_root, 'val', transform=get_transforms('val')
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        from model import LightweightFaceParser, verify_model_size
        model = verify_model_size()
        model = model.to(device)
        
        # Training config
        config = {
            'num_classes': 19,
            'lr': args.lr,
            'weight_decay': 1e-4,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }
        
        # Train
        from train import Trainer
        trainer = Trainer(model, train_loader, val_loader, device, config)
        trainer.train()
        
    elif args.mode == 'test':
        # Inference
        from inference import Inference, TestDataset
        from dataset import get_transforms
        
        test_dataset = TestDataset(
            Path(args.data_root) / 'test' / 'images',
            transform=get_transforms('val')
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )
        
        inference = Inference('best_model.pth', device)
        inference.run_inference(test_loader, 'test_results')

if __name__ == '__main__':
    main()