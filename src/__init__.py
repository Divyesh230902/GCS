"""
GCS: Graph-based Classification System for Medical Image Retrieval

A novel framework combining:
- SSM (State Space Model) for query handling
- CLIP embeddings for multimodal understanding  
- GraphRAG for graph-based retrieval and reasoning
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

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

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "CLIPEmbeddingExtractor": (".clip_embeddings", "CLIPEmbeddingExtractor"),
    "ImageEmbedding": (".clip_embeddings", "ImageEmbedding"),
    "MedicalKnowledgeGraph": (".graphRAG", "MedicalKnowledgeGraph"),
    "GraphRAGRetriever": (".graphRAG", "GraphRAGRetriever"),
    "GraphNode": (".graphRAG", "GraphNode"),
    "GraphEdge": (".graphRAG", "GraphEdge"),
    "SSMQueryProcessor": (".ssm", "SSMQueryProcessor"),
    "QueryResult": (".ssm", "QueryResult"),
    "MedicalImageDataset": (".data_utils", "MedicalImageDataset"),
    "create_data_loaders": (".data_utils", "create_data_loaders"),
    "get_dataset_info": (".data_utils", "get_dataset_info"),
    "get_model_config": (".model_config", "get_model_config"),
    "check_huggingface_auth": (".model_config", "check_huggingface_auth"),
    "get_available_models": (".model_config", "get_available_models"),
    "MedicalCommunityDetector": (".community_detection", "MedicalCommunityDetector"),
    "Community": (".community_detection", "Community"),
    "CommunitySummarizer": (".community_summarization", "CommunitySummarizer"),
    "CommunitySummary": (".community_summarization", "CommunitySummary"),
    "EnhancedGraphRAGRetriever": (".enhanced_graphrag", "EnhancedGraphRAGRetriever"),
    "EnhancedRetrievalResult": (".enhanced_graphrag", "EnhancedRetrievalResult"),
    "GraphRAGVisualizer": (".visualization", "GraphRAGVisualizer"),
    "GraphRAGEvaluator": (".evaluation", "GraphRAGEvaluator"),
    "AblationStudyEvaluator": (".evaluation", "AblationStudyEvaluator"),
    "GroundTruthGenerator": (".evaluation", "GroundTruthGenerator"),
    "RetrievalMetrics": (".evaluation", "RetrievalMetrics"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> Any:
    return sorted(set(list(globals().keys()) + list(__all__)))
