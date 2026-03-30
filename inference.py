import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import cv2
from model import LightweightFaceParser

class Inference:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        self.PALETTE = self._create_palette()
        
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model = LightweightFaceParser(num_classes=19)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        return model
    # PALETTE
    def _create_palette(self):
        PALETTE = np.array([[i, i, i] for i in range(256)])
        PALETTE[:19] = np.array([
        [0, 0, 0],       
        [255, 0, 0],     
        [0, 255, 255],   
        [0, 0, 255],     
        [0, 204, 0],     
        [128, 0, 128],   
        [255, 255, 0],   
        [0, 255, 0],     
        [255, 0, 255],   
        [128, 128, 0],   
        [165, 42, 42],   
        [255, 165, 0],   
        [210, 180, 140], 
        [255, 192, 203], 
        [128, 128, 128], 
        [255, 215, 0],   
        [255, 192, 203], 
        [255, 215, 0],   
        [128, 0, 128]    
        ])
        return PALETTE.reshape(-1).tolist()
    
    def _mask_to_palette_image(self, mask):
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.putpalette(self.PALETTE)
        return mask_img
    
    def predict(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0).to(self.device))
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
        return pred
    
    def visualize_prediction(self, image, prediction, save_path=None):
        """Overlay prediction on original image"""
        # mask
        palette_array = np.array(self.PALETTE).reshape(-1, 3)
        color_mask = palette_array[prediction].astype(np.uint8)  
    
         
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
    
        
        if image.shape[:2] != color_mask.shape[:2]:
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))
    
        # Blend with original image
        blended = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)
    
        if save_path:
            Image.fromarray(blended).save(save_path)
    
        return blended
    
    def run_inference(self, test_loader, output_dir):
        """Run inference on test dataset and save predictions"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
        # Create subdirectories
        masks_dir = output_dir / 'masks'
        vis_dir = output_dir / 'visualizations'
        masks_dir.mkdir(exist_ok=True)
        vis_dir.mkdir(exist_ok=True)
    
        print(f"Running inference on {len(test_loader.dataset)} images...")
    
        for batch_idx, (images, filenames) in enumerate(test_loader):
            for i, (image, filename) in enumerate(zip(images, filenames)):
                
                filename_path = Path(filename)
            
                # Get prediction
                pred = self.predict(image)
            
                # Save prediction mask with palette
                mask_path = masks_dir / f"{filename_path.stem}.png"
                mask_img = self._mask_to_palette_image(pred)
                mask_img.save(mask_path)
            
                # Save visualization 
                # Convert image tensor back to numpy
                img_np = ((image.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
                vis_path = vis_dir / f"{filename_path.stem}_vis.png"
                self.visualize_prediction(img_np, pred, vis_path)
            
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} images in batch {batch_idx + 1}")
    
        print(f"\nSaved predictions to {output_dir}")
        print(f"  - Masks: {masks_dir}")
        print(f"  - Visualizations: {vis_dir}")
    
        return {
            'num_images': len(test_loader.dataset),
            'output_dir': str(output_dir),
            'masks_dir': str(masks_dir),
            'visualizations_dir': str(vis_dir)
        }

# Test dataset for inference
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.image_files = sorted(list(self.test_dir.glob('*.jpg')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, img_path.name