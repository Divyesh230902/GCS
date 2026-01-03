"""
GraphRAG Framework for Medical Image Retrieval
Implements graph-based retrieval and reasoning over medical data
"""

import torch
import numpy as np
import networkx as nx
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import faiss
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from .clip_embeddings import CLIPEmbeddingExtractor, ImageEmbedding
from .ssm import SSMQueryProcessor, QueryResult

@dataclass
class GraphNode:
    """Node in the medical knowledge graph"""
    node_id: str
    node_type: str  # 'image', 'class', 'dataset', 'concept'
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    properties: Dict[str, Any] = None

@dataclass
class GraphEdge:
    """Edge in the medical knowledge graph"""
    source_id: str
    target_id: str
    edge_type: str  # 'belongs_to', 'similar_to', 'related_to', 'part_of'
    weight: float = 1.0
    metadata: Dict[str, Any] = None

@dataclass
class RetrievalResult:
    """Result from GraphRAG retrieval"""
    query: str
    retrieved_images: List[ImageEmbedding]
    reasoning_path: List[str]
    confidence_scores: List[float]
    graph_explanations: List[str]
    metadata: Dict[str, Any]

class MedicalKnowledgeGraph:
    """
    Medical knowledge graph for storing and reasoning over medical data
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_embeddings = {}
        self.edge_weights = {}
        self.clusters = {}
        self.concept_hierarchy = {}
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            embedding=node.embedding,
            metadata=node.metadata or {},
            properties=node.properties or {}
        )
        
        if node.embedding is not None:
            self.node_embeddings[node.node_id] = node.embedding
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph"""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type,
            weight=edge.weight,
            metadata=edge.metadata or {}
        )
        
        # Store edge weight for quick access
        edge_key = (edge.source_id, edge.target_id, edge.edge_type)
        self.edge_weights[edge_key] = edge.weight
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None) -> List[str]:
        """Get neighbors of a node, optionally filtered by edge types"""
        neighbors = []
        for neighbor in self.graph.graph.neighbors(node_id):
            if edge_types is None:
                neighbors.append(neighbor)
            else:
                # Check if any edge between nodes matches the edge types
                for edge_data in self.graph.graph[node_id][neighbor].values():
                    if edge_data.get('edge_type') in edge_types:
                        neighbors.append(neighbor)
                        break
        return neighbors
    
    def get_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Get all paths between two nodes up to max_length"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph.graph, source, target, cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def compute_node_centrality(self) -> Dict[str, float]:
        """Compute centrality measures for all nodes"""
        centrality = nx.betweenness_centrality(self.graph.graph)
        return centrality
    
    def find_similar_nodes(self, 
                          query_embedding: np.ndarray, 
                          node_type: Optional[str] = None,
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar nodes to query embedding"""
        similarities = []
        
        for node_id, embedding in self.node_embeddings.items():
            if embedding is None:
                continue
                
            # Filter by node type if specified
            if node_type and self.graph.graph.nodes[node_id].get('node_type') != node_type:
                continue
            
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append((node_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class GraphRAGRetriever:
    """
    GraphRAG-based retrieval system for medical images
    """
    
    def __init__(self, 
                 clip_extractor: CLIPEmbeddingExtractor,
                 ssm_processor: SSMQueryProcessor,
                 graph: MedicalKnowledgeGraph):
        """
        Initialize GraphRAG retriever
        
        Args:
            clip_extractor: CLIP embedding extractor
            ssm_processor: SSM query processor
            graph: Medical knowledge graph
        """
        self.clip_extractor = clip_extractor
        self.ssm_processor = ssm_processor
        self.graph = graph
        self.faiss_index = None
        self.image_embeddings = []
        self.node_embeddings = {}
        
        # Reasoning strategies
        self.reasoning_strategies = {
            "direct_similarity": self._direct_similarity_reasoning,
            "concept_expansion": self._concept_expansion_reasoning,
            "hierarchical_reasoning": self._hierarchical_reasoning,
            "path_based_reasoning": self._path_based_reasoning
        }
    
    def build_graph_from_embeddings(self, 
                                   image_embeddings: List[ImageEmbedding],
                                   text_embeddings: Dict[str, np.ndarray],
                                   cluster_threshold: float = 0.7):
        """
        Build the medical knowledge graph from embeddings
        
        Args:
            image_embeddings: List of image embeddings
            text_embeddings: Dictionary of text embeddings for classes
            cluster_threshold: Threshold for creating similarity edges
        """
        print("Building medical knowledge graph...")
        
        # Add image nodes
        for img_emb in image_embeddings:
            node = GraphNode(
                node_id=f"img_{len(self.image_embeddings)}",
                node_type="image",
                embedding=img_emb.embedding,
                metadata={
                    "image_path": img_emb.image_path,
                    "class_label": img_emb.class_label,
                    "dataset": img_emb.dataset
                }
            )
            self.graph.add_node(node)
            self.image_embeddings.append(img_emb)
            # Store embedding for similarity search
            self.node_embeddings[node.node_id] = img_emb.embedding
        
        # Add class nodes
        for class_name, text_emb in text_embeddings.items():
            node = GraphNode(
                node_id=f"class_{class_name}",
                node_type="class",
                embedding=text_emb,
                metadata={"class_name": class_name}
            )
            self.graph.add_node(node)
        
        # Add dataset nodes
        datasets = set(img_emb.dataset for img_emb in image_embeddings)
        for dataset in datasets:
            node = GraphNode(
                node_id=f"dataset_{dataset}",
                node_type="dataset",
                metadata={"dataset_name": dataset}
            )
            self.graph.add_node(node)
        
        # Create edges between images and their classes
        for i, img_emb in enumerate(image_embeddings):
            img_node_id = f"img_{i}"
            class_node_id = f"class_{img_emb.class_label}"
            dataset_node_id = f"dataset_{img_emb.dataset}"
            
            # Image belongs to class
            edge = GraphEdge(
                source_id=img_node_id,
                target_id=class_node_id,
                edge_type="belongs_to",
                weight=1.0
            )
            self.graph.add_edge(edge)
            
            # Image belongs to dataset
            edge = GraphEdge(
                source_id=img_node_id,
                target_id=dataset_node_id,
                edge_type="belongs_to",
                weight=1.0
            )
            self.graph.add_edge(edge)
        
        # Create similarity edges between similar images
        self._create_similarity_edges(cluster_threshold)
        
        # Create concept hierarchy
        self._build_concept_hierarchy()
        
        print(f"Graph built with {self.graph.graph.number_of_nodes()} nodes and {self.graph.graph.number_of_edges()} edges")
    
    def _create_similarity_edges(self, threshold: float):
        """Create edges between similar images"""
        print("Creating similarity edges...")
        
        # Get all image nodes
        image_nodes = [n for n in self.graph.graph.nodes() if n.startswith("img_")]
        
        for i, node1 in enumerate(image_nodes):
            for j, node2 in enumerate(image_nodes[i+1:], i+1):
                emb1 = self.graph.graph.nodes[node1]['embedding']
                emb2 = self.graph.graph.nodes[node2]['embedding']
                
                if emb1 is not None and emb2 is not None:
                    similarity = cosine_similarity(
                        emb1.reshape(1, -1),
                        emb2.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > threshold:
                        edge = GraphEdge(
                            source_id=node1,
                            target_id=node2,
                            edge_type="similar_to",
                            weight=similarity
                        )
                        self.graph.add_edge(edge)
    
    def _build_concept_hierarchy(self):
        """Build concept hierarchy based on medical relationships"""
        # Define medical concept relationships
        concept_relations = {
            "neurological_disorders": ["alzheimer", "parkinson", "ms"],
            "brain_tumors": ["glioma", "meningioma", "pituitary"],
            "dementia_types": ["mild_dementia", "moderate_dementia", "very_mild_dementia"],
            "normal_conditions": ["non_demented", "normal", "notumor"]
        }
        
        for parent_concept, child_concepts in concept_relations.items():
            # Add parent concept node
            parent_node = GraphNode(
                node_id=f"concept_{parent_concept}",
                node_type="concept",
                metadata={"concept_name": parent_concept}
            )
            self.graph.add_node(parent_node)
            
            # Connect to child concepts
            for child_concept in child_concepts:
                child_node_id = f"class_{child_concept}"
                if child_node_id in self.graph.graph.nodes:
                    edge = GraphEdge(
                        source_id=child_node_id,
                        target_id=f"concept_{parent_concept}",
                        edge_type="part_of",
                        weight=1.0
                    )
                    self.graph.add_edge(edge)
    
    def retrieve(self, 
                query: str, 
                top_k: int = 10,
                reasoning_strategy: str = "concept_expansion") -> RetrievalResult:
        """
        Retrieve relevant images using GraphRAG
        
        Args:
            query: User query
            top_k: Number of results to return
            reasoning_strategy: Strategy for graph-based reasoning
            
        Returns:
            RetrievalResult with retrieved images and reasoning
        """
        # Process query with SSM
        query_result = self.ssm_processor.process_query(query)
        
        # Generate query embedding
        query_embedding = self.clip_extractor.extract_text_embedding(query)
        
        # Apply reasoning strategy
        if reasoning_strategy in self.reasoning_strategies:
            reasoning_func = self.reasoning_strategies[reasoning_strategy]
            retrieved_images, reasoning_path, explanations = reasoning_func(
                query_embedding, query_result, top_k
            )
        else:
            # Default to direct similarity
            retrieved_images, reasoning_path, explanations = self._direct_similarity_reasoning(
                query_embedding, query_result, top_k
            )
        
        # Compute confidence scores
        confidence_scores = []
        for img_emb in retrieved_images:
            similarity = self.clip_extractor.compute_similarity(query_embedding, img_emb.embedding)
            confidence_scores.append(similarity)
        
        return RetrievalResult(
            query=query,
            retrieved_images=retrieved_images,
            reasoning_path=reasoning_path,
            confidence_scores=confidence_scores,
            graph_explanations=explanations,
            metadata={
                "reasoning_strategy": reasoning_strategy,
                "query_intent": query_result.intent,
                "query_confidence": query_result.confidence
            }
        )
    
    def _direct_similarity_reasoning(self, 
                                   query_embedding: np.ndarray,
                                   query_result: QueryResult,
                                   top_k: int) -> Tuple[List[ImageEmbedding], List[str], List[str]]:
        """Direct similarity-based reasoning"""
        # Find most similar images directly
        similarities = self.clip_extractor.find_similar_images(
            query_embedding, self.image_embeddings, top_k
        )
        
        retrieved_images = [img for img, _ in similarities]
        reasoning_path = ["direct_similarity"]
        explanations = [f"Found {len(retrieved_images)} images by direct similarity matching"]
        
        return retrieved_images, reasoning_path, explanations
    
    def _concept_expansion_reasoning(self, 
                                   query_embedding: np.ndarray,
                                   query_result: QueryResult,
                                   top_k: int) -> Tuple[List[ImageEmbedding], List[str], List[str]]:
        """Concept expansion-based reasoning"""
        # Find similar concept nodes
        similar_concepts = self.graph.find_similar_nodes(
            query_embedding, node_type="concept", top_k=5
        )
        
        retrieved_images = []
        reasoning_path = ["concept_expansion"]
        explanations = []
        
        for concept_id, similarity in similar_concepts:
            # Get images connected to this concept
            concept_images = self._get_images_for_concept(concept_id)
            retrieved_images.extend(concept_images)
            explanations.append(f"Expanded to concept {concept_id} (similarity: {similarity:.3f})")
        
        # Remove duplicates and limit results
        seen_paths = set()
        unique_images = []
        for img in retrieved_images:
            if img.image_path not in seen_paths:
                unique_images.append(img)
                seen_paths.add(img.image_path)
                if len(unique_images) >= top_k:
                    break
        
        return unique_images[:top_k], reasoning_path, explanations
    
    def _hierarchical_reasoning(self, 
                              query_embedding: np.ndarray,
                              query_result: QueryResult,
                              top_k: int) -> Tuple[List[ImageEmbedding], List[str], List[str]]:
        """Hierarchical reasoning through concept hierarchy"""
        # Start with most similar class
        similar_classes = self.graph.find_similar_nodes(
            query_embedding, node_type="class", top_k=3
        )
        
        retrieved_images = []
        reasoning_path = ["hierarchical_reasoning"]
        explanations = []
        
        for class_id, similarity in similar_classes:
            # Get images for this class
            class_images = self._get_images_for_class(class_id)
            retrieved_images.extend(class_images)
            
            # Find parent concepts
            parent_concepts = self.graph.get_neighbors(class_id, ["part_of"])
            for parent in parent_concepts:
                parent_images = self._get_images_for_concept(parent)
                retrieved_images.extend(parent_images)
                explanations.append(f"Hierarchical expansion: {class_id} -> {parent}")
        
        # Remove duplicates and limit
        seen_paths = set()
        unique_images = []
        for img in retrieved_images:
            if img.image_path not in seen_paths:
                unique_images.append(img)
                seen_paths.add(img.image_path)
                if len(unique_images) >= top_k:
                    break
        
        return unique_images[:top_k], reasoning_path, explanations
    
    def _path_based_reasoning(self, 
                            query_embedding: np.ndarray,
                            query_result: QueryResult,
                            top_k: int) -> Tuple[List[ImageEmbedding], List[str], List[str]]:
        """Path-based reasoning through graph traversal"""
        # Find starting nodes (most similar to query)
        similar_nodes = self.graph.find_similar_nodes(query_embedding, top_k=5)
        
        retrieved_images = []
        reasoning_path = ["path_based_reasoning"]
        explanations = []
        
        for start_node, similarity in similar_nodes:
            # Find paths to image nodes
            image_nodes = [n for n in self.graph.graph.nodes() if n.startswith("img_")]
            
            for img_node in image_nodes:
                paths = self.graph.get_paths(start_node, img_node, max_length=3)
                if paths:
                    # Path found, add image
                    img_idx = int(img_node.split("_")[1])
                    if img_idx < len(self.image_embeddings):
                        retrieved_images.append(self.image_embeddings[img_idx])
                        explanations.append(f"Path found: {start_node} -> {img_node}")
        
        # Remove duplicates and limit
        seen_paths = set()
        unique_images = []
        for img in retrieved_images:
            if img.image_path not in seen_paths:
                unique_images.append(img)
                seen_paths.add(img.image_path)
                if len(unique_images) >= top_k:
                    break
        
        return unique_images[:top_k], reasoning_path, explanations
    
    def _get_images_for_concept(self, concept_id: str) -> List[ImageEmbedding]:
        """Get all images connected to a concept"""
        images = []
        concept_images = self.graph.get_neighbors(concept_id, ["belongs_to"])
        
        for img_node in concept_images:
            if img_node.startswith("img_"):
                img_idx = int(img_node.split("_")[1])
                if img_idx < len(self.image_embeddings):
                    images.append(self.image_embeddings[img_idx])
        
        return images
    
    def _get_images_for_class(self, class_id: str) -> List[ImageEmbedding]:
        """Get all images for a specific class"""
        images = []
        class_images = self.graph.get_neighbors(class_id, ["belongs_to"])
        
        for img_node in class_images:
            if img_node.startswith("img_"):
                img_idx = int(img_node.split("_")[1])
                if img_idx < len(self.image_embeddings):
                    images.append(self.image_embeddings[img_idx])
        
        return images
    
    def save_graph(self, filepath: str):
        """Save the graph to file"""
        graph_data = {
            "nodes": dict(self.graph.graph.nodes(data=True)),
            "edges": list(self.graph.graph.edges(data=True)),
            "image_embeddings": self.image_embeddings
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph_data, f)
    
    def load_graph(self, filepath: str):
        """Load the graph from file"""
        with open(filepath, 'rb') as f:
            graph_data = pickle.load(f)
        
        # Rebuild graph
        self.graph.clear()
        for node_id, node_data in graph_data["nodes"].items():
            self.graph.add_node(node_id, **node_data)
        
        for source, target, edge_data in graph_data["edges"]:
            self.graph.add_edge(source, target, **edge_data)
        
        self.image_embeddings = graph_data["image_embeddings"]

# Example usage and testing
if __name__ == "__main__":
    # Initialize components
    clip_extractor = CLIPEmbeddingExtractor()
    ssm_processor = SSMQueryProcessor()
    graph = MedicalKnowledgeGraph()
    
    # Initialize GraphRAG retriever
    retriever = GraphRAGRetriever(clip_extractor, ssm_processor, graph)
    
    print("GraphRAG retriever initialized successfully!")
    print("Ready to build knowledge graph and perform retrievals.")
