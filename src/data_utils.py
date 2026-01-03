"""
Data preprocessing utilities for medical image datasets
Supports: Alzheimer's, Brain Tumor, Parkinson's, and Multiple Sclerosis datasets
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json

class MedicalImageDataset(Dataset):
    """Unified dataset class for all medical image datasets"""
    
    def __init__(self, 
                 data_dir: str, 
                 dataset_name: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 max_samples_per_class: Optional[int] = None):
        """
        Args:
            data_dir: Path to data directory
            dataset_name: One of ['alzheimer', 'brain_tumor', 'parkinson', 'ms']
            split: 'train' or 'test' (only applicable for brain_tumor)
            transform: Image transformations
            max_samples_per_class: Limit samples per class for faster experimentation
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform or self._get_default_transform()
        self.max_samples_per_class = max_samples_per_class
        
        self.samples, self.classes = self._load_dataset()
        
    def _get_default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_dataset(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load dataset based on dataset_name"""
        if self.dataset_name == 'alzheimer':
            return self._load_alzheimer()
        elif self.dataset_name == 'brain_tumor':
            return self._load_brain_tumor()
        elif self.dataset_name == 'parkinson':
            return self._load_parkinson()
        elif self.dataset_name == 'ms':
            return self._load_ms()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_alzheimer(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load Alzheimer's dataset"""
        base_path = os.path.join(self.data_dir, 'AlzheimerDataset')
        classes = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit samples if specified
            if self.max_samples_per_class:
                class_files = class_files[:self.max_samples_per_class]
            
            for file_name in class_files:
                file_path = os.path.join(class_path, file_name)
                samples.append((file_path, class_to_idx[class_name]))
        
        return samples, classes
    
    def _load_brain_tumor(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load Brain Tumor dataset"""
        base_path = os.path.join(self.data_dir, 'brain-tumor-mri-dataset', self.split.title())
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit samples if specified
            if self.max_samples_per_class:
                class_files = class_files[:self.max_samples_per_class]
            
            for file_name in class_files:
                file_path = os.path.join(class_path, file_name)
                samples.append((file_path, class_to_idx[class_name]))
        
        return samples, classes
    
    def _load_parkinson(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load Parkinson's dataset"""
        base_path = os.path.join(self.data_dir, 'parkinsons_dataset_processed')
        classes = ['normal', 'parkinson']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit samples if specified
            if self.max_samples_per_class:
                class_files = class_files[:self.max_samples_per_class]
            
            for file_name in class_files:
                file_path = os.path.join(class_path, file_name)
                samples.append((file_path, class_to_idx[class_name]))
        
        return samples, classes
    
    def _load_ms(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load Multiple Sclerosis dataset"""
        base_path = os.path.join(self.data_dir, 'ms_slices_central')
        classes = ['Normal', 'MS']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit samples if specified
            if self.max_samples_per_class:
                class_files = class_files[:self.max_samples_per_class]
            
            for file_name in class_files:
                file_path = os.path.join(class_path, file_name)
                samples.append((file_path, class_to_idx[class_name]))
        
        return samples, classes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        image = Image.open(file_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, file_path  # Return file_path for potential use in retrieval
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution in the dataset"""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

def create_data_loaders(data_dir: str, 
                       dataset_name: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       max_samples_per_class: Optional[int] = None) -> Dict[str, DataLoader]:
    """Create data loaders for training and testing"""
    
    # For brain tumor, we have separate train/test splits
    if dataset_name == 'brain_tumor':
        train_dataset = MedicalImageDataset(
            data_dir, dataset_name, 'train', 
            max_samples_per_class=max_samples_per_class
        )
        test_dataset = MedicalImageDataset(
            data_dir, dataset_name, 'test',
            max_samples_per_class=max_samples_per_class
        )
    else:
        # For other datasets, we'll split them ourselves
        full_dataset = MedicalImageDataset(
            data_dir, dataset_name, 
            max_samples_per_class=max_samples_per_class
        )
        
        # Simple 80/20 split
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'test': test_loader,
        'classes': train_dataset.classes if hasattr(train_dataset, 'classes') else test_dataset.classes
    }

def get_dataset_info(data_dir: str) -> Dict[str, Dict]:
    """Get information about all available datasets"""
    datasets = ['alzheimer', 'brain_tumor', 'parkinson', 'ms']
    info = {}
    
    for dataset_name in datasets:
        try:
            if dataset_name == 'brain_tumor':
                train_dataset = MedicalImageDataset(data_dir, dataset_name, 'train')
                test_dataset = MedicalImageDataset(data_dir, dataset_name, 'test')
                info[dataset_name] = {
                    'train_samples': len(train_dataset),
                    'test_samples': len(test_dataset),
                    'total_samples': len(train_dataset) + len(test_dataset),
                    'classes': train_dataset.classes,
                    'train_class_distribution': train_dataset.get_class_distribution(),
                    'test_class_distribution': test_dataset.get_class_distribution()
                }
            else:
                dataset = MedicalImageDataset(data_dir, dataset_name)
                info[dataset_name] = {
                    'total_samples': len(dataset),
                    'classes': dataset.classes,
                    'class_distribution': dataset.get_class_distribution()
                }
        except Exception as e:
            info[dataset_name] = {'error': str(e)}
    
    return info

if __name__ == "__main__":
    # Test the data loading
    data_dir = "/Users/divyeshpatel/Desktop/Buffalo/course/Fall2025/research/GCS/data"
    
    print("Dataset Information:")
    info = get_dataset_info(data_dir)
    for dataset_name, dataset_info in info.items():
        print(f"\n{dataset_name.upper()}:")
        if 'error' in dataset_info:
            print(f"  Error: {dataset_info['error']}")
        else:
            print(f"  Total samples: {dataset_info.get('total_samples', 'N/A')}")
            print(f"  Classes: {dataset_info.get('classes', 'N/A')}")
            if 'class_distribution' in dataset_info:
                print(f"  Class distribution: {dataset_info['class_distribution']}")
