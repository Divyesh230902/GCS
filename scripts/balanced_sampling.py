"""
Balanced Sampling Strategy for Medical Datasets
Creates equal distribution of each disease and normal cases to reduce bias
"""

import os
import random
import shutil
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import json
from data_utils import get_dataset_info

class BalancedSampler:
    """
    Creates balanced datasets with equal distribution of disease and normal cases
    """
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str = "./balanced_data",
                 samples_per_class: int = 500,
                 random_seed: int = 42):
        """
        Initialize balanced sampler
        
        Args:
            data_dir: Path to original data directory
            output_dir: Path to save balanced datasets
            samples_per_class: Number of samples per class (disease/normal)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.samples_per_class = samples_per_class
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            "alzheimer": {
                "disease_classes": ["Mild Dementia", "Moderate Dementia", "Very mild Dementia"],
                "normal_class": "Non Demented",
                "source_path": "AlzheimerDataset"
            },
            "brain_tumor": {
                "disease_classes": ["glioma", "meningioma", "pituitary"],
                "normal_class": "notumor",
                "source_path": "brain-tumor-mri-dataset"
            },
            "parkinson": {
                "disease_classes": ["parkinson"],
                "normal_class": "normal",
                "source_path": "parkinsons_dataset_processed"
            },
            "ms": {
                "disease_classes": ["MS"],
                "normal_class": "Normal",
                "source_path": "ms_slices_central"
            }
        }
    
    def create_balanced_dataset(self, dataset_name: str) -> Dict[str, int]:
        """
        Create balanced dataset for a specific disease
        
        Args:
            dataset_name: Name of the dataset (alzheimer, brain_tumor, parkinson, ms)
            
        Returns:
            Dictionary with sampling statistics
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        source_path = os.path.join(self.data_dir, config["source_path"])
        output_path = os.path.join(self.output_dir, f"balanced_{dataset_name}")
        
        print(f"\nCreating balanced dataset for {dataset_name.upper()}")
        print(f"Source: {source_path}")
        print(f"Output: {output_path}")
        print(f"Samples per class: {self.samples_per_class}")
        
        # Create output directory structure
        os.makedirs(output_path, exist_ok=True)
        
        sampling_stats = {}
        
        if dataset_name == "brain_tumor":
            # Special handling for brain tumor (has train/test split)
            return self._create_balanced_brain_tumor(source_path, output_path)
        
        # For other datasets, sample from all available data
        all_classes = config["disease_classes"] + [config["normal_class"]]
        
        for class_name in all_classes:
            class_source_path = os.path.join(source_path, class_name)
            class_output_path = os.path.join(output_path, class_name)
            
            if not os.path.exists(class_source_path):
                print(f"Warning: Class {class_name} not found in {class_source_path}")
                sampling_stats[class_name] = 0
                continue
            
            # Get all files in the class
            all_files = [f for f in os.listdir(class_source_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sample files
            if len(all_files) >= self.samples_per_class:
                sampled_files = random.sample(all_files, self.samples_per_class)
            else:
                # If not enough files, use all available files
                sampled_files = all_files
                print(f"Warning: Only {len(all_files)} files available for {class_name}, using all")
            
            # Create output directory for this class
            os.makedirs(class_output_path, exist_ok=True)
            
            # Copy sampled files
            for file_name in sampled_files:
                src_file = os.path.join(class_source_path, file_name)
                dst_file = os.path.join(class_output_path, file_name)
                shutil.copy2(src_file, dst_file)
            
            sampling_stats[class_name] = len(sampled_files)
            print(f"  {class_name}: {len(sampled_files)} samples")
        
        return sampling_stats
    
    def _create_balanced_brain_tumor(self, source_path: str, output_path: str) -> Dict[str, int]:
        """Special handling for brain tumor dataset with train/test split"""
        sampling_stats = {}
        
        for split in ["Training", "Testing"]:
            split_source = os.path.join(source_path, split)
            split_output = os.path.join(output_path, split)
            os.makedirs(split_output, exist_ok=True)
            
            print(f"  Processing {split} split...")
            
            # Get all classes
            all_classes = ["glioma", "meningioma", "notumor", "pituitary"]
            
            for class_name in all_classes:
                class_source_path = os.path.join(split_source, class_name)
                class_output_path = os.path.join(split_output, class_name)
                
                if not os.path.exists(class_source_path):
                    print(f"    Warning: Class {class_name} not found in {class_source_path}")
                    continue
                
                # Get all files in the class
                all_files = [f for f in os.listdir(class_source_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # Sample files (use half for each split)
                samples_for_split = self.samples_per_class // 2
                if len(all_files) >= samples_for_split:
                    sampled_files = random.sample(all_files, samples_for_split)
                else:
                    sampled_files = all_files
                
                # Create output directory for this class
                os.makedirs(class_output_path, exist_ok=True)
                
                # Copy sampled files
                for file_name in sampled_files:
                    src_file = os.path.join(class_source_path, file_name)
                    dst_file = os.path.join(class_output_path, file_name)
                    shutil.copy2(src_file, dst_file)
                
                # Update stats
                key = f"{split}_{class_name}"
                sampling_stats[key] = len(sampled_files)
                print(f"    {class_name}: {len(sampled_files)} samples")
        
        return sampling_stats
    
    def create_all_balanced_datasets(self) -> Dict[str, Dict[str, int]]:
        """
        Create balanced datasets for all available datasets
        
        Returns:
            Dictionary with sampling statistics for all datasets
        """
        print("=" * 60)
        print("CREATING BALANCED MEDICAL DATASETS")
        print("=" * 60)
        
        all_stats = {}
        
        for dataset_name in self.dataset_configs.keys():
            try:
                stats = self.create_balanced_dataset(dataset_name)
                all_stats[dataset_name] = stats
            except Exception as e:
                print(f"Error creating balanced dataset for {dataset_name}: {e}")
                all_stats[dataset_name] = {"error": str(e)}
        
        # Save sampling statistics
        stats_file = os.path.join(self.output_dir, "sampling_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"\nSampling statistics saved to: {stats_file}")
        return all_stats
    
    def get_balanced_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about the created balanced datasets
        
        Returns:
            Dictionary with balanced dataset information
        """
        balanced_data_dir = self.output_dir
        datasets = ['alzheimer', 'brain_tumor', 'parkinson', 'ms']
        info = {}
        
        for dataset_name in datasets:
            try:
                if dataset_name == 'brain_tumor':
                    # Special handling for brain tumor
                    train_path = os.path.join(balanced_data_dir, f"balanced_{dataset_name}", "Training")
                    test_path = os.path.join(balanced_data_dir, f"balanced_{dataset_name}", "Testing")
                    
                    if os.path.exists(train_path) and os.path.exists(test_path):
                        train_samples = sum(len(os.listdir(os.path.join(train_path, d))) 
                                          for d in os.listdir(train_path) 
                                          if os.path.isdir(os.path.join(train_path, d)))
                        test_samples = sum(len(os.listdir(os.path.join(test_path, d))) 
                                         for d in os.listdir(test_path) 
                                         if os.path.isdir(os.path.join(test_path, d)))
                        
                        info[dataset_name] = {
                            'total_samples': train_samples + test_samples,
                            'train_samples': train_samples,
                            'test_samples': test_samples,
                            'samples_per_class': self.samples_per_class // 2
                        }
                    else:
                        info[dataset_name] = {'error': 'Dataset not found'}
                else:
                    # Other datasets
                    dataset_path = os.path.join(balanced_data_dir, f"balanced_{dataset_name}")
                    if os.path.exists(dataset_path):
                        total_samples = sum(len(os.listdir(os.path.join(dataset_path, d))) 
                                          for d in os.listdir(dataset_path) 
                                          if os.path.isdir(os.path.join(dataset_path, d)))
                        info[dataset_name] = {
                            'total_samples': total_samples,
                            'samples_per_class': self.samples_per_class
                        }
                    else:
                        info[dataset_name] = {'error': 'Dataset not found'}
            except Exception as e:
                info[dataset_name] = {'error': str(e)}
        
        return info
    
    def create_balanced_data_utils(self):
        """
        Create a modified data_utils.py that works with balanced datasets
        """
        balanced_data_utils = '''
"""
Modified data_utils.py for balanced datasets
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json

class BalancedMedicalImageDataset(Dataset):
    """Dataset class for balanced medical image datasets"""
    
    def __init__(self, 
                 data_dir: str, 
                 dataset_name: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            data_dir: Path to balanced data directory
            dataset_name: One of ['alzheimer', 'brain_tumor', 'parkinson', 'ms']
            split: 'train' or 'test' (only applicable for brain_tumor)
            transform: Image transformations
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform or self._get_default_transform()
        
        self.samples, self.classes = self._load_balanced_dataset()
        
    def _get_default_transform(self):
        """Default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_balanced_dataset(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load balanced dataset"""
        if self.dataset_name == 'alzheimer':
            return self._load_balanced_alzheimer()
        elif self.dataset_name == 'brain_tumor':
            return self._load_balanced_brain_tumor()
        elif self.dataset_name == 'parkinson':
            return self._load_balanced_parkinson()
        elif self.dataset_name == 'ms':
            return self._load_balanced_ms()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_balanced_alzheimer(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load balanced Alzheimer's dataset"""
        base_path = os.path.join(self.data_dir, 'balanced_alzheimer')
        classes = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file_name in class_files:
                file_path = os.path.join(class_path, file_name)
                samples.append((file_path, class_to_idx[class_name]))
        
        return samples, classes
    
    def _load_balanced_brain_tumor(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load balanced Brain Tumor dataset"""
        base_path = os.path.join(self.data_dir, 'balanced_brain_tumor', self.split.title())
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file_name in class_files:
                file_path = os.path.join(class_path, file_name)
                samples.append((file_path, class_to_idx[class_name]))
        
        return samples, classes
    
    def _load_balanced_parkinson(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load balanced Parkinson's dataset"""
        base_path = os.path.join(self.data_dir, 'balanced_parkinson')
        classes = ['normal', 'parkinson']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file_name in class_files:
                file_path = os.path.join(class_path, file_name)
                samples.append((file_path, class_to_idx[class_name]))
        
        return samples, classes
    
    def _load_balanced_ms(self) -> Tuple[List[Tuple[str, int]], List[str]]:
        """Load balanced Multiple Sclerosis dataset"""
        base_path = os.path.join(self.data_dir, 'balanced_ms')
        classes = ['Normal', 'MS']
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(base_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            class_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
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
        
        return image, label, file_path
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution in the dataset"""
        class_counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

def create_balanced_data_loaders(data_dir: str, 
                                dataset_name: str,
                                batch_size: int = 32,
                                num_workers: int = 4) -> Dict[str, DataLoader]:
    """Create data loaders for balanced datasets"""
    
    if dataset_name == 'brain_tumor':
        train_dataset = BalancedMedicalImageDataset(data_dir, dataset_name, 'train')
        test_dataset = BalancedMedicalImageDataset(data_dir, dataset_name, 'test')
    else:
        # For other datasets, split them ourselves
        full_dataset = BalancedMedicalImageDataset(data_dir, dataset_name)
        
        # 80/20 split
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

def get_balanced_dataset_info(data_dir: str) -> Dict[str, Dict]:
    """Get information about balanced datasets"""
    datasets = ['alzheimer', 'brain_tumor', 'parkinson', 'ms']
    info = {}
    
    for dataset_name in datasets:
        try:
            if dataset_name == 'brain_tumor':
                train_dataset = BalancedMedicalImageDataset(data_dir, dataset_name, 'train')
                test_dataset = BalancedMedicalImageDataset(data_dir, dataset_name, 'test')
                info[dataset_name] = {
                    'train_samples': len(train_dataset),
                    'test_samples': len(test_dataset),
                    'total_samples': len(train_dataset) + len(test_dataset),
                    'classes': train_dataset.classes,
                    'train_class_distribution': train_dataset.get_class_distribution(),
                    'test_class_distribution': test_dataset.get_class_distribution()
                }
            else:
                dataset = BalancedMedicalImageDataset(data_dir, dataset_name)
                info[dataset_name] = {
                    'total_samples': len(dataset),
                    'classes': dataset.classes,
                    'class_distribution': dataset.get_class_distribution()
                }
        except Exception as e:
            info[dataset_name] = {'error': str(e)}
    
    return info
'''
        
        # Write the balanced data utils
        with open(os.path.join(self.output_dir, 'balanced_data_utils.py'), 'w') as f:
            f.write(balanced_data_utils)
        
        print(f"Balanced data utilities created at: {os.path.join(self.output_dir, 'balanced_data_utils.py')}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize balanced sampler
    data_dir = "/Users/divyeshpatel/Desktop/Buffalo/course/Fall2025/research/GCS/data"
    sampler = BalancedSampler(
        data_dir=data_dir,
        output_dir="./balanced_data",
        samples_per_class=500,  # 500 samples per class
        random_seed=42
    )
    
    # Create all balanced datasets
    print("Creating balanced datasets...")
    stats = sampler.create_all_balanced_datasets()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("BALANCED DATASET STATISTICS")
    print("=" * 60)
    
    for dataset_name, dataset_stats in stats.items():
        print(f"\n{dataset_name.upper()}:")
        if 'error' in dataset_stats:
            print(f"  Error: {dataset_stats['error']}")
        else:
            for class_name, count in dataset_stats.items():
                print(f"  {class_name}: {count} samples")
    
    # Get balanced dataset info
    print("\n" + "=" * 60)
    print("BALANCED DATASET INFO")
    print("=" * 60)
    
    balanced_info = sampler.get_balanced_dataset_info()
    for dataset_name, info in balanced_info.items():
        print(f"\n{dataset_name.upper()}:")
        if 'error' in info:
            print(f"  Error: {info['error']}")
        else:
            print(f"  Total samples: {info.get('total_samples', 'N/A')}")
            if 'samples_per_class' in info:
                print(f"  Samples per class: {info['samples_per_class']}")
    
    # Create balanced data utilities
    sampler.create_balanced_data_utils()
    
    print("\n✅ Balanced datasets created successfully!")
    print("✅ All classes now have equal representation!")
    print("✅ Bias reduced through balanced sampling!")
