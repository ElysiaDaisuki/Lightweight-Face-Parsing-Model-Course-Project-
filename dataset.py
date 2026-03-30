import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FaceParsingDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        if split in ['train', 'val']:
            self.images_dir = self.root_dir / split / 'images'
            self.masks_dir = self.root_dir / split / 'masks'
            self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        else:  # test
            self.images_dir = self.root_dir / split / 'images'
            self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.split in ['train', 'val']:
            mask_path = self.masks_dir / (img_path.stem + '.png')
            mask = np.array(Image.open(mask_path))
            
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
                mask = mask.long()
            return image, mask
        else:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            return image, img_path.name

# Data Augmentation
def get_transforms(split='train'):
    if split == 'train':
        return A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    else:  # val or test
        return A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])