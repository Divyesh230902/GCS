"""
CLIP Embeddings for Medical Image Retrieval
Extracts multimodal embeddings from medical images and text descriptions
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoProcessor
from PIL import Image
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pickle
from tqdm import tqdm
import faiss
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class ImageEmbedding:
    """Container for image embedding data"""
    image_path: str
    embedding: np.ndarray
    class_label: str
    dataset: str
    metadata: Dict[str, any]

class CLIPEmbeddingExtractor:
    """
    CLIP-based embedding extractor for medical images
    """
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch32",  # Public, no auth needed
                 device: str = "auto",
                 cache_dir: str = "./embeddings_cache"):
        """
        Initialize CLIP embedding extractor
        
        Args:
            model_name: Hugging Face CLIP model name
            device: Device to run on
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load CLIP model
        self._load_model()
        
        # Medical text descriptions for different conditions
        self.medical_descriptions = {
            "alzheimer": {
                "Mild Dementia": "Brain scan showing mild cognitive decline with slight hippocampal atrophy",
                "Moderate Dementia": "MRI scan displaying moderate Alzheimer's disease with significant brain shrinkage",
                "Non Demented": "Normal healthy brain scan with no signs of dementia or cognitive impairment",
                "Very mild Dementia": "Early stage Alzheimer's disease with minimal cognitive changes visible"
            },
            "brain_tumor": {
                "glioma": "MRI scan showing glioma brain tumor with irregular mass and surrounding edema",
                "meningioma": "Brain scan displaying meningioma tumor with well-defined borders and dural attachment",
                "notumor": "Normal brain MRI scan with no evidence of tumors or abnormal masses",
                "pituitary": "Pituitary gland tumor visible in brain scan with sellar region involvement"
            },
            "parkinson": {
                "normal": "Healthy brain scan showing normal dopamine-producing regions",
                "parkinson": "Brain scan indicating Parkinson's disease with reduced dopamine activity"
            },
            "ms": {
                "Normal": "Normal brain MRI with no signs of multiple sclerosis lesions",
                "MS": "Brain scan showing multiple sclerosis lesions with white matter abnormalities"
            }
        }
    
    def _load_model(self):
        """Load CLIP model using Hugging Face transformers"""
        try:
            print(f"Loading Hugging Face CLIP model: {self.model_name}")
            
            # Try multiple approaches to load CLIP
            try:
                # Approach 1: Standard CLIPModel (preferred for embeddings)
                from transformers import CLIPModel, CLIPProcessor
                self.model = CLIPModel.from_pretrained(self.model_name, trust_remote_code=True)
                self.processor = CLIPProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception as e1:
                print(f"Standard CLIP loading failed: {e1}")
                try:
                    # Approach 2: Use AutoModel
                    from transformers import AutoModel, AutoProcessor
                    self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                    self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                except Exception as e2:
                    print(f"AutoModel loading failed: {e2}")
                    # Will fall through to fallback
                    raise e2
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"⚠️  All CLIP loading methods failed")
            print(f"Error: {str(e)[:200]}")
            print("\n→ Using fallback embedding system (rule-based)")
            print("→ For real embeddings, try:")
            print("   1. pip install --upgrade transformers")
            print("   2. Or use offline mode with pre-downloaded models")
            
            # Fallback to simple embedding system
            self.model = None
            self.processor = None
    
    def extract_image_embedding(self, 
                              image_path: str, 
                              class_label: str, 
                              dataset: str,
                              metadata: Optional[Dict] = None) -> ImageEmbedding:
        """
        Extract CLIP embedding from a single image
        
        Args:
            image_path: Path to the image file
            class_label: Class label for the image
            dataset: Dataset name (alzheimer, brain_tumor, etc.)
            metadata: Additional metadata
            
        Returns:
            ImageEmbedding object
        """
        if self.model is None or self.processor is None:
            # Fallback to simple embedding
            embedding = self._simple_image_embedding(image_path, class_label, dataset)
        else:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process image with CLIP processor
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Extract image features using the dedicated image method
            with torch.no_grad():
                # Use get_image_features for image-only processing
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embedding = image_features.cpu().numpy().flatten()
        
        return ImageEmbedding(
            image_path=image_path,
            embedding=embedding,
            class_label=class_label,
            dataset=dataset,
            metadata=metadata or {}
        )
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract CLIP embedding from text
        
        Args:
            text: Input text description
            
        Returns:
            Text embedding as numpy array
        """
        if self.model is None or self.processor is None:
            # Fallback to simple text embedding
            return self._simple_text_embedding(text)
        
        # Process text with CLIP processor
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        # Extract text features using the dedicated text method
        with torch.no_grad():
            # Use get_text_features for text-only processing
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().flatten()
    
    def _simple_image_embedding(self, image_path: str, class_label: str, dataset: str) -> np.ndarray:
        """Simple fallback image embedding based on metadata"""
        # Create embedding based on class and dataset
        embedding = np.zeros(512)  # Standard embedding size
        
        # Encode class information
        class_hash = hash(class_label) % 100
        embedding[class_hash] = 1.0
        
        # Encode dataset information
        dataset_hash = hash(dataset) % 100
        embedding[100 + dataset_hash] = 1.0
        
        # Encode file path information
        path_hash = hash(os.path.basename(image_path)) % 100
        embedding[200 + path_hash] = 1.0
        
        # Add some random features to make embeddings unique
        np.random.seed(hash(image_path) % 2**32)
        embedding[300:] = np.random.random(212)
        
        return embedding
    
    def _simple_text_embedding(self, text: str) -> np.ndarray:
        """Simple fallback text embedding based on keywords"""
        text_lower = text.lower()
        embedding = np.zeros(512)
        
        # Medical keywords mapping
        medical_keywords = {
            "alzheimer": 0, "dementia": 1, "brain": 2, "tumor": 3, "glioma": 4,
            "meningioma": 5, "pituitary": 6, "parkinson": 7, "sclerosis": 8,
            "ms": 9, "normal": 10, "mri": 11, "scan": 12, "lesion": 13,
            "atrophy": 14, "cognitive": 15, "neurological": 16
        }
        
        # Encode keywords found in text
        for keyword, idx in medical_keywords.items():
            if keyword in text_lower:
                embedding[idx] = 1.0
        
        # Encode text length and other features
        embedding[50] = min(len(text.split()) / 20.0, 1.0)  # Normalized word count
        embedding[51] = 1.0 if "disease" in text_lower else 0.0
        embedding[52] = 1.0 if "disorder" in text_lower else 0.0
        embedding[53] = 1.0 if "abnormal" in text_lower else 0.0
        
        # Add some random features
        np.random.seed(hash(text) % 2**32)
        embedding[100:] = np.random.random(412)
        
        return embedding
    
    def batch_extract_embeddings(self, 
                                image_paths: List[str],
                                class_labels: List[str],
                                dataset: str,
                                batch_size: int = 32,
                                save_cache: bool = True) -> List[ImageEmbedding]:
        """
        Extract embeddings for a batch of images
        
        Args:
            image_paths: List of image file paths
            class_labels: List of corresponding class labels
            dataset: Dataset name
            batch_size: Batch size for processing
            save_cache: Whether to save embeddings to cache
            
        Returns:
            List of ImageEmbedding objects
        """
        embeddings = []
        cache_file = os.path.join(self.cache_dir, f"{dataset}_embeddings.pkl")
        
        # Check if cached embeddings exist
        if os.path.exists(cache_file) and save_cache:
            print(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data) == len(image_paths):
                    return cached_data
        
        print(f"Extracting embeddings for {len(image_paths)} images...")
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = class_labels[i:i+batch_size]
            
            # Load and preprocess batch
            batch_images = []
            valid_indices = []
            
            for j, path in enumerate(batch_paths):
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            if self.model is None or self.processor is None:
                # Use fallback for each image
                for j, idx in enumerate(valid_indices):
                    embedding = self._simple_image_embedding(
                        batch_paths[idx], batch_labels[idx], dataset
                    )
                    embeddings.append(ImageEmbedding(
                        image_path=batch_paths[idx],
                        embedding=embedding,
                        class_label=batch_labels[idx],
                        dataset=dataset,
                        metadata={}
                    ))
                continue
            
            # Process images with CLIP processor
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Use get_image_features for batch image processing
                batch_features = self.model.get_image_features(**inputs)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            
            # Create ImageEmbedding objects
            for j, idx in enumerate(valid_indices):
                embedding = ImageEmbedding(
                    image_path=batch_paths[idx],
                    embedding=batch_features[j].cpu().numpy(),
                    class_label=batch_labels[idx],
                    dataset=dataset,
                    metadata={}
                )
                embeddings.append(embedding)
        
        # Save to cache
        if save_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Embeddings cached to {cache_file}")
        
        return embeddings
    
    def create_text_embeddings_for_dataset(self, dataset: str) -> Dict[str, np.ndarray]:
        """
        Create text embeddings for all classes in a dataset
        
        Args:
            dataset: Dataset name
            
        Returns:
            Dictionary mapping class names to text embeddings
        """
        if dataset not in self.medical_descriptions:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        text_embeddings = {}
        descriptions = self.medical_descriptions[dataset]
        
        for class_name, description in descriptions.items():
            embedding = self.extract_text_embedding(description)
            text_embeddings[class_name] = embedding
        
        return text_embeddings
    
    def compute_similarity(self, 
                          image_embedding: np.ndarray, 
                          text_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between image and text embeddings
        
        Args:
            image_embedding: Image embedding vector
            text_embedding: Text embedding vector
            
        Returns:
            Cosine similarity score
        """
        return cosine_similarity(
            image_embedding.reshape(1, -1), 
            text_embedding.reshape(1, -1)
        )[0][0]
    
    def find_similar_images(self, 
                           query_embedding: np.ndarray,
                           image_embeddings: List[ImageEmbedding],
                           top_k: int = 10) -> List[Tuple[ImageEmbedding, float]]:
        """
        Find most similar images to a query embedding
        
        Args:
            query_embedding: Query embedding vector
            image_embeddings: List of image embeddings to search
            top_k: Number of top similar images to return
            
        Returns:
            List of (ImageEmbedding, similarity_score) tuples
        """
        similarities = []
        
        for img_emb in image_embeddings:
            similarity = self.compute_similarity(query_embedding, img_emb.embedding)
            similarities.append((img_emb, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def create_faiss_index(self, 
                          image_embeddings: List[ImageEmbedding],
                          index_type: str = "flat") -> faiss.Index:
        """
        Create FAISS index for efficient similarity search
        
        Args:
            image_embeddings: List of image embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            
        Returns:
            FAISS index
        """
        if not image_embeddings:
            raise ValueError("No embeddings provided")
        
        # Get embedding dimension
        dim = len(image_embeddings[0].embedding)
        
        # Create index based on type
        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 100)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Prepare embeddings for indexing
        embeddings_matrix = np.array([emb.embedding for emb in image_embeddings])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Train and add to index
        if index_type == "ivf":
            index.train(embeddings_matrix)
        
        index.add(embeddings_matrix)
        
        return index
    
    def search_faiss_index(self, 
                          index: faiss.Index,
                          query_embedding: np.ndarray,
                          image_embeddings: List[ImageEmbedding],
                          top_k: int = 10) -> List[Tuple[ImageEmbedding, float]]:
        """
        Search FAISS index for similar images
        
        Args:
            index: FAISS index
            query_embedding: Query embedding
            image_embeddings: Original image embeddings list
            top_k: Number of results to return
            
        Returns:
            List of (ImageEmbedding, similarity_score) tuples
        """
        # Normalize query embedding
        query_normalized = query_embedding.copy()
        faiss.normalize_L2(query_normalized.reshape(1, -1))
        
        # Search
        scores, indices = index.search(query_normalized.reshape(1, -1), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(image_embeddings):
                results.append((image_embeddings[idx], float(score)))
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Initialize CLIP extractor
    extractor = CLIPEmbeddingExtractor()
    
    # Test with a sample image (if available)
    data_dir = "/Users/divyeshpatel/Desktop/Buffalo/course/Fall2025/research/GCS/data"
    
    # Test text embedding extraction
    print("Testing text embedding extraction:")
    test_text = "Brain scan showing signs of Alzheimer's disease with hippocampal atrophy"
    text_embedding = extractor.extract_text_embedding(test_text)
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Test medical descriptions
    print("\nTesting medical text descriptions:")
    for dataset in ["alzheimer", "brain_tumor", "parkinson", "ms"]:
        print(f"\n{dataset.upper()}:")
        text_embeddings = extractor.create_text_embeddings_for_dataset(dataset)
        for class_name, embedding in text_embeddings.items():
            print(f"  {class_name}: {embedding.shape}")
    
    print("\nCLIP embedding extractor ready!")
