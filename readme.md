# Lightweight Face Parsing Model
A lightweight deep learning model for face parsing (semantic segmentation of facial components) with optimized architecture using depthwise separable convolutions and residual blocks. According to the course requirements, designed to stay under 1.82 million parameters while maintaining good performance. Under these parameter constraints, the model can operate on low-computing-power devices.
## Contents 
- [Overview](#overview) - [Project Structure](#project-structure) - [Requirements](#requirements) - [Dataset Preparation](#dataset-preparation) - [Training](#training) - [Inference](#inference) - [Model Architecture](#model-architecture) - [License](#license) 
## Overview 
This project implements a lightweight face parsing model that segments facial regions into 16 different classes (e.g., eyes, nose, mouth, hair). Key features: - Depthwise separable convolutions for parameter efficiency - Residual blocks for better feature propagation - Auto train-validation split of training data - Data augmentation for training robustness - Inference pipeline with visualization of segmentation masks - Strict parameter limit (<= 1,821,085) for course requirements
<div style="display:flex; gap:8px;">
<img src="https://github.com/ElysiaDaisuki/Lightweight-Face-Parsing-Model-Course-Project-/raw/main/assets/example_ori.jpg" width="32%">
<img src="https://github.com/ElysiaDaisuki/Lightweight-Face-Parsing-Model-Course-Project-/raw/main/assets/example_vis.png" width="32%">
<img src="https://github.com/ElysiaDaisuki/Lightweight-Face-Parsing-Model-Course-Project-/raw/main/assets/example.png" width="32%">
</div>

 ## Project Structure
 solution/

├── dataset_setup.py # Dataset splitting (train/val) and directory setup

├── dataset.py # Custom Dataset class and data transforms

├── model.py # Lightweight face parsing model architecture

├── main.py # Main entry point (training/inference)

├── inference.py # Inference pipeline and visualization

└── train.py # Training loop implementation
  
## Requirements 
Install the required dependencies: 
```pip install -r requirements.txt```
### Minimum Requirements

-   Python 3.8+
-   PyTorch 1.9+
-   CUDA (optional, for GPU acceleration)
-   8GB RAM (16GB recommended for batch training)

## Dataset Preparation

### Dataset Structure

data_root/   
├── train/  
│                ├── images/ # JPG images   
│ └── masks/ # Corresponding PNG masks (same filename as images)   
└── test/   
│                ├──images/ # JPG images for inference  
### Automatic Train-Validation Split

The `prepare_dataset_structure` function automatically splits the training data into 90% training and 10% validation, creating a `val` directory with images/masks.
## Training

Run the training script with required arguments:
```
python main.py --mode train --data_root /path/to/dataset --batch_size 8 --epochs 100 --lr 1e-3
```

### Training Arguments
| Argument | Description | Default |
|----------|-------------|---------| 
| `--mode` | Operation mode (train/test) | Required | 
| `--data_root` | Path to root dataset directory | Required |
 | `--batch_size` | Training batch size | 8 | 
 | `--epochs` | Number of training epochs | 100 |
 | `--lr` | Learning rate | 0.001 |
 ### Training Features

-   Data augmentation (Gaussian noise, brightness/contrast adjustment, rotation)
-   Normalization (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
-   Automatic model size verification (ensures <1.82M parameters)
-   Separate validation loop for monitoring performance
## Inference

Run inference on test images and generate segmentation masks + visualizations:
```
python solution/main.py --mode test --data_root /path/to/dataset
```
### Inference Output

The script creates a `test_results` directory with:  
test_results/   
├── masks/ # Segmentation masks (PNG with color palette)   
└── visualizations/ # Overlay of masks on original images (blended)  

## Model Architecture
The `LightweightFaceParser` model consists of:

1.  **Input Layer**: 3→64 convolution with batch normalization and ReLU
2.  **Encoder**: 4 residual blocks (depthwise separable convolutions) with downsampling (stride=2)
    
    -   64→64 → 64→128 → 128→256 → 256→512
    -   Stores skip connections at each stage for decoder fusion
    
3.  **Decoder**: 3 upsampling stages with concatenation of encoder skip features
    
    -   Bilinear upsampling (scale=2) + concatenation with skip features
    -   Depthwise separable convolutions for feature fusion and channel reduction
    -   512+256→256 → 256+128→128 → 128+64→64
    
4.  **Final Upsampling**: Bilinear upsampling (scale=2) to restore input resolution
5.  **Output Layer**: 64→19 convolution for class prediction

### Parameter Efficiency

-   Depthwise separable convolutions reduce parameters by ~8-9x compared to standard convolutions
-   Residual shortcuts preserve feature flow without excessive parameters
-   Model includes automatic parameter counting to ensure compliance with 1.82M parameter limit
## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Notes

-   The `train.py` file contains the full training loop with loss calculation (CrossEntropyLoss), optimizer (AdamW), and validation metrics (IoU, accuracy).
-   Ensure mask files have the same base filename as their corresponding images (e.g., `face1.jpg` → `face1.png`).
-   The model uses 19 facial classes - adjust the `num_classes` parameter if using a different number of segmentation classes.
