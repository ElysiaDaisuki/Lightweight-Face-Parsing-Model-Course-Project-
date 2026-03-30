import os
import shutil
import random
from pathlib import Path

def prepare_dataset_structure(data_root):

    train_path = Path(data_root) / 'train'
    test_path = Path(data_root) / 'test'
    
    # Create validation directories
    val_path = Path(data_root) / 'val'
    val_path.mkdir(exist_ok=True)
    (val_path / 'images').mkdir(exist_ok=True)
    (val_path / 'masks').mkdir(exist_ok=True)
    
    # Get all training image-mask pairs
    train_images = sorted((train_path / 'images').glob('*.jpg'))
    
    # Split 90-10 for train-val
    random.seed(11)
    random.shuffle(train_images)
    split_idx = int(0.9 * len(train_images))
    
    val_images = train_images[split_idx:]
    train_images = train_images[:split_idx]
    
    # Move validation images and their masks
    for img_path in val_images:
        # Move image
        shutil.move(str(img_path), str(val_path / 'images' / img_path.name))
        
        # Move corresponding mask
        mask_name = img_path.stem + '.png'  
        mask_path = train_path / 'masks' / mask_name
        if mask_path.exists():
            shutil.move(str(mask_path), str(val_path / 'masks' / mask_name))
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    return train_path, val_path, test_path