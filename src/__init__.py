"""
GCS: Graph-based Classification System for Medical Image Retrieval

A novel framework combining:
- SSM (State Space Model) for query handling
- CLIP embeddings for multimodal understanding  
- GraphRAG for graph-based retrieval and reasoning
"""

from .clip_embeddings import CLIPEmbeddingExtractor, ImageEmbedding
from .graphRAG import MedicalKnowledgeGraph, GraphRAGRetriever, GraphNode, GraphEdge
from .ssm import SSMQueryProcessor, QueryResult
from .data_utils import MedicalImageDataset, create_data_loaders, get_dataset_info
from .model_config import get_model_config, check_huggingface_auth, get_available_models
from .community_detection import MedicalCommunityDetector, Community
from .community_summarization import CommunitySummarizer, CommunitySummary
from .enhanced_graphrag import EnhancedGraphRAGRetriever, EnhancedRetrievalResult
from .visualization import GraphRAGVisualizer
from .evaluation import GraphRAGEvaluator, AblationStudyEvaluator, GroundTruthGenerator, RetrievalMetrics

__version__ = "1.0.0"
__author__ = "GCS Research Team"

__all__ = [
    "CLIPEmbeddingExtractor",
    "ImageEmbedding", 
    "MedicalKnowledgeGraph",
    "GraphRAGRetriever",
    "GraphNode",
    "GraphEdge",
    "SSMQueryProcessor",
    "QueryResult",
    "MedicalImageDataset",
    "create_data_loaders",
    "get_dataset_info",
    "get_model_config",
    "check_huggingface_auth",
    "get_available_models",
    "MedicalCommunityDetector",
    "Community",
    "CommunitySummarizer",
    "CommunitySummary",
    "EnhancedGraphRAGRetriever",
    "EnhancedRetrievalResult",
    "GraphRAGVisualizer",
    "GraphRAGEvaluator",
    "AblationStudyEvaluator",
    "GroundTruthGenerator",
    "RetrievalMetrics"
]
