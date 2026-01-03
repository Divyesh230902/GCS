"""
Enhanced GraphRAG Retriever with Microsoft GraphRAG concepts
Implements Global/Local/Hybrid search modes for medical images
"""

import numpy as np
import torch
import os
import pickle
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from .graphRAG import GraphRAGRetriever, MedicalKnowledgeGraph, RetrievalResult, ImageEmbedding
from .community_detection import MedicalCommunityDetector, Community
from .community_summarization import CommunitySummarizer, CommunitySummary
from .ssm import SSMQueryProcessor, QueryResult
from .clip_embeddings import CLIPEmbeddingExtractor

@dataclass
class EnhancedRetrievalResult:
    """Enhanced retrieval result with hierarchical context"""
    query: str
    search_mode: str  # 'global', 'local', 'hybrid'
    retrieved_images: List[ImageEmbedding]
    reasoning_path: List[str]
    explanations: List[str]
    community_context: List[CommunitySummary]
    confidence: float
    metadata: Dict

class EnhancedGraphRAGRetriever:
    """
    Microsoft GraphRAG-inspired retriever for medical images
    Implements hierarchical community-based retrieval
    """
    
    def __init__(self,
                 clip_extractor: CLIPEmbeddingExtractor,
                 ssm_processor: SSMQueryProcessor,
                 graph: MedicalKnowledgeGraph):
        """
        Initialize enhanced GraphRAG retriever
        
        Args:
            clip_extractor: CLIP embedding extractor
            ssm_processor: SSM query processor
            graph: Medical knowledge graph
        """
        self.clip_extractor = clip_extractor
        self.ssm_processor = ssm_processor
        self.graph = graph
        
        # Community detection and summarization
        self.community_detector = MedicalCommunityDetector()
        self.community_summarizer = CommunitySummarizer(ssm_processor)
        
        # Storage
        self.communities = {}
        self.community_summaries = {}
        self.image_embeddings = []
        self.node_embeddings = {}
        self.node_metadata = {}
        
        print("Enhanced GraphRAG Retriever initialized with community support")
    
    def build_enhanced_graph(self,
                            image_embeddings: List[ImageEmbedding],
                            text_embeddings: Dict[str, np.ndarray] = None):
        """
        Build enhanced graph with hierarchical communities
        
        Args:
            image_embeddings: List of image embeddings
            text_embeddings: Optional text embeddings for concepts
        """
        print("\n" + "=" * 60)
        print("BUILDING ENHANCED MEDICAL KNOWLEDGE GRAPH")
        print("=" * 60)
        
        # Store embeddings
        self.image_embeddings = image_embeddings
        
        # Step 1: Build basic graph structure
        print("\nStep 1: Building base graph structure...")
        self._build_base_graph(image_embeddings, text_embeddings or {})
        
        # Step 2: Detect hierarchical communities
        print("\nStep 2: Detecting hierarchical communities...")
        self.communities = self.community_detector.detect_communities(
            self.graph.graph,
            self.node_embeddings,
            self.node_metadata,
            max_levels=3
        )
        
        # Step 3: Generate community summaries
        print("\nStep 3: Generating community summaries...")
        self.community_summaries = self.community_summarizer.summarize_communities(
            self.communities,
            self.node_metadata
        )
        
        # Step 4: Update graph with community information
        print("\nStep 4: Updating graph with community memberships...")
        self._update_graph_with_communities()
        
        print("\n✅ Enhanced graph built successfully!")
        print(f"   - Nodes: {self.graph.graph.number_of_nodes()}")
        print(f"   - Edges: {self.graph.graph.number_of_edges()}")
        print(f"   - Communities: {len(self.communities)}")
        print(f"   - Summaries: {len(self.community_summaries)}")
        print("=" * 60)

    def save(self, filepath: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
        """Persist a built enhanced graph + retrieval state for team testing."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        state = {
            "artifact_version": 1,
            "image_embeddings": self.image_embeddings,
            "node_embeddings": self.node_embeddings,
            "node_metadata": self.node_metadata,
            "graph": self.graph.graph,
            "communities": self.communities,
            "community_summaries": self.community_summaries,
            "extra": extra or {},
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(
        cls,
        filepath: str,
        *,
        clip_extractor: CLIPEmbeddingExtractor,
        ssm_processor: SSMQueryProcessor,
        graph: Optional[MedicalKnowledgeGraph] = None,
    ) -> Tuple["EnhancedGraphRAGRetriever", Dict[str, Any]]:
        """Load a persisted enhanced graph + retrieval state."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        instance = cls(clip_extractor=clip_extractor, ssm_processor=ssm_processor, graph=graph or MedicalKnowledgeGraph())
        instance.image_embeddings = state.get("image_embeddings", [])
        instance.node_embeddings = state.get("node_embeddings", {})
        instance.node_metadata = state.get("node_metadata", {})
        instance.communities = state.get("communities", {})
        instance.community_summaries = state.get("community_summaries", {})
        instance.graph.graph = state.get("graph", instance.graph.graph)
        return instance, state.get("extra", {})

    def _build_base_graph(self,
                         image_embeddings: List[ImageEmbedding],
                         text_embeddings: Dict[str, np.ndarray]):
        """Build base graph structure"""
        # Add image nodes
        for img_emb in image_embeddings:
            node_id = f"img_{len(self.node_embeddings)}"
            
            # Store embedding and metadata
            self.node_embeddings[node_id] = img_emb.embedding
            self.node_metadata[node_id] = {
                'image_path': img_emb.image_path,
                'class_label': img_emb.class_label,
                'dataset': img_emb.dataset,
                'metadata': img_emb.metadata
            }
            
            # Add to graph
            self.graph.graph.add_node(
                node_id,
                node_type='image',
                embedding=img_emb.embedding,
                **self.node_metadata[node_id]
            )
        
        # Create similarity edges
        print("  Creating similarity edges...")
        self._create_similarity_edges(threshold=0.7)
        
        print(f"  ✓ Added {len(self.node_embeddings)} nodes")
    
    def _create_similarity_edges(self, threshold: float = 0.7):
        """Create edges based on embedding similarity"""
        node_ids = list(self.node_embeddings.keys())
        embeddings = np.array([self.node_embeddings[nid] for nid in node_ids])
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Add edges for high similarity pairs
        edge_count = 0
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                if similarities[i, j] > threshold:
                    self.graph.graph.add_edge(
                        node_ids[i],
                        node_ids[j],
                        edge_type='similar',
                        weight=float(similarities[i, j])
                    )
                    edge_count += 1
        
        print(f"  ✓ Added {edge_count} similarity edges")
    
    def _update_graph_with_communities(self):
        """Update graph nodes with community membership"""
        for comm_id, community in self.communities.items():
            for node_id in community.member_nodes:
                if self.graph.graph.has_node(node_id):
                    # Add community information to node
                    node_data = self.graph.graph.nodes[node_id]
                    if 'communities' not in node_data:
                        node_data['communities'] = []
                    node_data['communities'].append(comm_id)
                    node_data[f'community_L{community.level}'] = comm_id
    
    def retrieve(self,
                query: str,
                top_k: int = 10,
                search_mode: str = "auto",
                query_embedding: Optional[np.ndarray] = None) -> EnhancedRetrievalResult:
        """
        Enhanced retrieval with multiple search modes
        
        Args:
            query: User query
            top_k: Number of results to return
            search_mode: 'global', 'local', 'hybrid', or 'auto'
            
        Returns:
            Enhanced retrieval result
        """
        print(f"\n🔍 Processing query: '{query}'")
        print(f"   Search mode: {search_mode}")
        
        # Process query with SSM
        query_result = self.ssm_processor.process_query(query or "image query")

        # Generate query embedding if not provided (text-only default)
        if query_embedding is None:
            query_embedding = self.clip_extractor.extract_text_embedding(query)
        
        # Determine search mode if auto
        if search_mode == "auto":
            search_mode = self._determine_search_mode(query_result)
            print(f"   Auto-selected mode: {search_mode}")
        
        # Route to appropriate search strategy
        if search_mode == "global":
            result = self._global_search(query, query_result, top_k, query_embedding=query_embedding)
        elif search_mode == "local":
            result = self._local_search(query, query_result, top_k, query_embedding=query_embedding)
        else:  # hybrid
            result = self._hybrid_search(query, query_result, top_k, query_embedding=query_embedding)
        
        print(f"✅ Retrieved {len(result.retrieved_images)} results")
        return result
    
    def _determine_search_mode(self, query_result: QueryResult) -> str:
        """Determine appropriate search mode based on query"""
        intent = query_result.intent
        
        # Global search for broad, analytical queries
        if intent in ['analysis', 'comparison', 'pattern_detection']:
            return 'global'
        
        # Local search for specific, retrieval queries
        elif intent in ['retrieval', 'similarity_search']:
            return 'local'
        
        # Hybrid for classification and complex queries
        else:
            return 'hybrid'
    
    def _global_search(self,
                      query: str,
                      query_result: QueryResult,
                      top_k: int,
                      *,
                      query_embedding: np.ndarray) -> EnhancedRetrievalResult:
        """
        Global search: Use community summaries for broad understanding
        Microsoft GraphRAG map-reduce approach
        """
        print("   Using GLOBAL search (community-based)...")

        # Step 1: Find relevant communities
        relevant_communities = self._find_relevant_communities(query_embedding, max_communities=5)
        
        # Step 2: Map phase - get insights from each community
        community_contexts = []
        for comm_id in relevant_communities:
            if comm_id in self.community_summaries:
                community_contexts.append(self.community_summaries[comm_id])
        
        # Step 3: Reduce phase - aggregate and select representative cases
        retrieved_images = self._select_representative_from_communities(
            relevant_communities, query_embedding, top_k
        )
        
        # Generate reasoning path
        reasoning_path = [
            "Global search across community summaries",
            f"Analyzed {len(relevant_communities)} relevant communities",
            f"Selected {len(retrieved_images)} representative cases"
        ]
        
        # Generate explanations
        explanations = [
            f"Community {comm.community_id}: {comm.summary_text}"
            for comm in community_contexts[:3]
        ]
        
        return EnhancedRetrievalResult(
            query=query,
            search_mode='global',
            retrieved_images=retrieved_images,
            reasoning_path=reasoning_path,
            explanations=explanations,
            community_context=community_contexts,
            confidence=0.85,
            metadata={'communities_analyzed': len(relevant_communities)}
        )
    
    def _local_search(self,
                     query: str,
                     query_result: QueryResult,
                     top_k: int,
                     *,
                     query_embedding: np.ndarray) -> EnhancedRetrievalResult:
        """
        Local search: Direct similarity search with entity-level details
        """
        print("   Using LOCAL search (entity-based)...")
        
        # Direct similarity search
        similarities = []
        for node_id, embedding in self.node_embeddings.items():
            if node_id.startswith('img_'):
                sim = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0, 0]
                similarities.append((node_id, sim))
        
        # Sort and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_nodes = similarities[:top_k]
        
        # Convert to ImageEmbedding objects
        retrieved_images = []
        for node_id, sim in top_nodes:
            metadata = self.node_metadata[node_id]
            merged_meta = dict(metadata.get("metadata", {}) or {})
            if "user_tags" in metadata:
                merged_meta["user_tags"] = metadata["user_tags"]
            merged_meta["similarity"] = sim
            retrieved_images.append(ImageEmbedding(
                image_path=metadata['image_path'],
                embedding=self.node_embeddings[node_id],
                class_label=metadata['class_label'],
                dataset=metadata['dataset'],
                metadata=merged_meta
            ))
        
        # Generate reasoning path
        reasoning_path = [
            "Local search using direct similarity matching",
            f"Computed similarity for {len(self.node_embeddings)} nodes",
            f"Selected top-{top_k} most similar cases"
        ]
        
        # Generate explanations
        explanations = [
            f"Case {i+1}: {img.class_label} from {img.dataset} (similarity: {img.metadata['similarity']:.3f})"
            for i, img in enumerate(retrieved_images[:5])
        ]
        
        return EnhancedRetrievalResult(
            query=query,
            search_mode='local',
            retrieved_images=retrieved_images,
            reasoning_path=reasoning_path,
            explanations=explanations,
            community_context=[],
            confidence=0.90,
            metadata={'total_candidates': len(self.node_embeddings)}
        )
    
    def _hybrid_search(self,
                      query: str,
                      query_result: QueryResult,
                      top_k: int,
                      *,
                      query_embedding: np.ndarray) -> EnhancedRetrievalResult:
        """
        Hybrid search: Combine global context with local precision
        """
        print("   Using HYBRID search (combined approach)...")
        
        # Get both global and local results
        global_result = self._global_search(query, query_result, top_k // 2, query_embedding=query_embedding)
        local_result = self._local_search(query, query_result, top_k // 2, query_embedding=query_embedding)
        
        # Combine results (deduplicate)
        combined_images = global_result.retrieved_images + local_result.retrieved_images
        seen_paths = set()
        unique_images = []
        for img in combined_images:
            if img.image_path not in seen_paths:
                unique_images.append(img)
                seen_paths.add(img.image_path)
        
        # Take top-k
        retrieved_images = unique_images[:top_k]
        
        # Combine reasoning and explanations
        reasoning_path = [
            "Hybrid search combining global and local strategies",
            *global_result.reasoning_path[1:2],
            *local_result.reasoning_path[1:2]
        ]
        
        explanations = [
            "Global Context:",
            *global_result.explanations[:2],
            "Local Matches:",
            *local_result.explanations[:2]
        ]
        
        return EnhancedRetrievalResult(
            query=query,
            search_mode='hybrid',
            retrieved_images=retrieved_images,
            reasoning_path=reasoning_path,
            explanations=explanations,
            community_context=global_result.community_context,
            confidence=0.87,
            metadata={'global_k': len(global_result.retrieved_images), 'local_k': len(local_result.retrieved_images)}
        )
    
    def _find_relevant_communities(self,
                                  query_embedding: np.ndarray,
                                  max_communities: int = 5) -> List[str]:
        """Find most relevant communities for a query"""
        similarities = []
        
        for comm_id, community in self.communities.items():
            # Compute similarity with community centroid
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                community.centroid_embedding.reshape(1, -1)
            )[0, 0]
            similarities.append((comm_id, sim))
        
        # Sort and return top communities
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [comm_id for comm_id, _ in similarities[:max_communities]]
    
    def _select_representative_from_communities(self,
                                               community_ids: List[str],
                                               query_embedding: np.ndarray,
                                               top_k: int) -> List[ImageEmbedding]:
        """Select representative images from communities"""
        candidates = []
        
        # Collect candidates from communities
        for comm_id in community_ids:
            if comm_id in self.communities:
                community = self.communities[comm_id]
                for node_id in community.member_nodes[:20]:  # Limit per community
                    if node_id in self.node_embeddings:
                        embedding = self.node_embeddings[node_id]
                        sim = cosine_similarity(
                            query_embedding.reshape(1, -1),
                            embedding.reshape(1, -1)
                        )[0, 0]
                        candidates.append((node_id, sim))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to ImageEmbedding objects
        retrieved_images = []
        for node_id, sim in candidates[:top_k]:
            metadata = self.node_metadata[node_id]
            merged_meta = dict(metadata.get("metadata", {}) or {})
            if "user_tags" in metadata:
                merged_meta["user_tags"] = metadata["user_tags"]
            merged_meta["similarity"] = sim
            retrieved_images.append(ImageEmbedding(
                image_path=metadata['image_path'],
                embedding=self.node_embeddings[node_id],
                class_label=metadata['class_label'],
                dataset=metadata['dataset'],
                metadata=merged_meta
            ))
        
        return retrieved_images
    
    def get_node_hierarchical_context(self, node_id: str) -> Dict:
        """Get hierarchical context for a node"""
        context = {
            'node_id': node_id,
            'metadata': self.node_metadata.get(node_id, {}),
            'communities': [],
            'summaries': []
        }
        
        # Find node's communities at each level
        for level in [0, 1, 2]:
            for comm_id, community in self.communities.items():
                if community.level == level and node_id in community.member_nodes:
                    context['communities'].append(comm_id)
                    if comm_id in self.community_summaries:
                        summary = self.community_summaries[comm_id]
                        context['summaries'].append({
                            'level': level,
                            'text': summary.summary_text,
                            'findings': summary.key_findings
                        })
                    break
        
        return context
    
    def print_retrieval_summary(self, result: EnhancedRetrievalResult):
        """Print detailed retrieval summary"""
        print("\n" + "=" * 60)
        print("ENHANCED RETRIEVAL RESULT")
        print("=" * 60)
        print(f"Query: {result.query}")
        print(f"Search Mode: {result.search_mode.upper()}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"\nReasoning Path:")
        for step in result.reasoning_path:
            print(f"  • {step}")
        print(f"\nTop Explanations:")
        for exp in result.explanations[:5]:
            print(f"  • {exp}")
        print(f"\nRetrieved Images: {len(result.retrieved_images)}")
        for i, img in enumerate(result.retrieved_images[:5], 1):
            sim = img.metadata.get('similarity', 'N/A')
            print(f"  {i}. {img.class_label} ({img.dataset}) - sim: {sim}")
        print("=" * 60)


if __name__ == "__main__":
    print("Enhanced GraphRAG Retriever")
    print("Microsoft GraphRAG-inspired implementation for medical images")
    print("Features: Global/Local/Hybrid search with hierarchical communities")
