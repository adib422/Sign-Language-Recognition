import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2


# CONFIGURATION

DATASET_PATH = "asl_dataset"
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 6

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cpu')

# ALBUMENTATIONS TRANSFORMS (GRAYSCALE)

def get_train_transform():
    return A.Compose([
           A.Resize(IMG_SIZE, IMG_SIZE),
           A.Rotate(limit=15, p=0.5),
           A.HorizontalFlip(p=0.3), 
           A.Normalize(mean=[0.5], std=[0.5]),
           ToTensorV2()
       ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

# DATASET CLASS

class ASLDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, self.labels[idx]

# MAIN PIPELINE

def prepare_data():
    
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path '{DATASET_PATH}' not found!")
    
    class_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()])
    class_to_idx = {f.name: idx for idx, f in enumerate(class_folders)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    image_paths = []
    labels = []
    
    for class_folder in tqdm(class_folders, desc="Loading dataset"):
        class_name = class_folder.name
        class_idx = class_to_idx[class_name]
        
        image_files = list(class_folder.glob("*.jpg")) + \
                     list(class_folder.glob("*.jpeg")) + \
                     list(class_folder.glob("*.png"))
        
        for img_path in image_files:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    print(f"Total images: {len(image_paths):,}")
    
    # Split dataset: train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=TEST_RATIO, 
        random_state=RANDOM_SEED, stratify=labels
    )
    
    val_ratio_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adj,
        random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Create datasets with Albumentations transforms
    train_dataset = ASLDataset(X_train, y_train, get_train_transform())
    val_dataset = ASLDataset(X_val, y_val, get_val_transform())
    test_dataset = ASLDataset(X_test, y_test, get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"Batch size: {BATCH_SIZE} | Train batches: {len(train_loader)}")
    
    # Save metadata
    metadata = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_to_idx)
    }
    torch.save(metadata, 'dataset_metadata.pt')
    
    # Visualize samples
    images, labels_batch = next(iter(train_loader))
    
    print(f"Image shape: {images.shape} (grayscale)")
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Denormalize
            img = images[i] * 0.5 + 0.5
            img = img.squeeze().numpy().clip(0, 1)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{idx_to_class[labels_batch[i].item()]}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_batch.png', dpi=150, bbox_inches='tight')
    print("Saved sample_batch.png")
    plt.close()
    
    return train_loader, val_loader, test_loader, metadata

if __name__ == "__main__":
    
    train_loader, val_loader, test_loader, metadata = prepare_data()
    
    print("\n✅ Dataset Preparartion Complete")
