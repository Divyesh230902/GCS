"""
Community Detection for Medical Knowledge Graph
Implements Leiden-inspired hierarchical clustering for medical images
Based on disease type and visual features
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import torch
from collections import defaultdict

@dataclass
class Community:
    """Represents a community in the hierarchical structure"""
    id: str
    level: int
    parent_id: Optional[str]
    member_nodes: List[str]
    disease_type: str
    description: str
    centroid_embedding: np.ndarray
    metadata: Dict

class MedicalCommunityDetector:
    """
    Hierarchical community detection for medical images
    Combines disease classification with visual similarity
    """
    
    def __init__(self, 
                 disease_weight: float = 0.6,
                 visual_weight: float = 0.4,
                 min_community_size: int = 3):
        """
        Initialize community detector
        
        Args:
            disease_weight: Weight for disease-based clustering
            visual_weight: Weight for visual similarity clustering
            min_community_size: Minimum nodes per community
        """
        self.disease_weight = disease_weight
        self.visual_weight = visual_weight
        self.min_community_size = min_community_size
        self.communities = {}
        self.hierarchy = {}
        
    def detect_communities(self,
                          graph: nx.Graph,
                          node_embeddings: Dict[str, np.ndarray],
                          node_metadata: Dict[str, Dict],
                          max_levels: int = 3) -> Dict[str, Community]:
        """
        Detect hierarchical communities in medical knowledge graph
        
        Args:
            graph: NetworkX graph
            node_embeddings: Dictionary of node embeddings
            node_metadata: Dictionary of node metadata
            max_levels: Maximum hierarchy levels
            
        Returns:
            Dictionary of communities by ID
        """
        print(f"Detecting hierarchical communities (max {max_levels} levels)...")
        
        # Level 0: Disease Type Communities (Global)
        print("Level 0: Disease type clustering...")
        level_0_communities = self._cluster_by_disease_type(
            graph, node_embeddings, node_metadata
        )
        
        # Level 1: Visual Feature Communities (Mid)
        print("Level 1: Visual feature clustering...")
        level_1_communities = self._cluster_by_visual_features(
            graph, node_embeddings, node_metadata, level_0_communities
        )
        
        # Level 2: Fine-grained Communities (Local)
        print("Level 2: Fine-grained clustering...")
        level_2_communities = self._cluster_fine_grained(
            graph, node_embeddings, node_metadata, level_1_communities
        )
        
        # Combine all communities
        all_communities = {}
        all_communities.update(level_0_communities)
        all_communities.update(level_1_communities)
        all_communities.update(level_2_communities)
        
        self.communities = all_communities
        self._build_hierarchy()
        
        print(f"✅ Detected {len(all_communities)} communities across {max_levels} levels")
        return all_communities
    
    def _cluster_by_disease_type(self,
                                 graph: nx.Graph,
                                 node_embeddings: Dict[str, np.ndarray],
                                 node_metadata: Dict[str, Dict]) -> Dict[str, Community]:
        """Level 0: Cluster by disease type"""
        communities = {}
        disease_groups = defaultdict(list)
        
        # Group nodes by disease type
        for node_id in graph.nodes():
            if node_id.startswith('img_'):
                metadata = node_metadata.get(node_id, {})
                disease = metadata.get('dataset', 'unknown')
                disease_groups[disease].append(node_id)
        
        # Create communities
        for idx, (disease, members) in enumerate(disease_groups.items()):
            if len(members) >= self.min_community_size:
                comm_id = f"L0_C{idx}_{disease}"
                
                # Compute centroid embedding
                embeddings = [node_embeddings[node] for node in members if node in node_embeddings]
                centroid = np.mean(embeddings, axis=0) if embeddings else np.zeros(512)
                
                communities[comm_id] = Community(
                    id=comm_id,
                    level=0,
                    parent_id=None,
                    member_nodes=members,
                    disease_type=disease,
                    description=f"Global community: {disease} dataset",
                    centroid_embedding=centroid,
                    metadata={'disease': disease, 'size': len(members)}
                )
        
        return communities
    
    def _cluster_by_visual_features(self,
                                    graph: nx.Graph,
                                    node_embeddings: Dict[str, np.ndarray],
                                    node_metadata: Dict[str, Dict],
                                    parent_communities: Dict[str, Community]) -> Dict[str, Community]:
        """Level 1: Cluster by visual features within disease types"""
        communities = {}
        comm_idx = 0
        
        # For each parent community, cluster by visual similarity
        for parent_id, parent_comm in parent_communities.items():
            members = parent_comm.member_nodes
            
            if len(members) < self.min_community_size * 2:
                continue  # Too small to subdivide
            
            # Get embeddings
            embeddings = []
            valid_members = []
            for node in members:
                if node in node_embeddings:
                    embeddings.append(node_embeddings[node])
                    valid_members.append(node)
            
            if len(embeddings) < self.min_community_size * 2:
                continue
            
            embeddings = np.array(embeddings)
            
            # Determine optimal number of clusters
            n_clusters = min(max(2, len(valid_members) // 10), 5)
            
            # Agglomerative clustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(embeddings)
            
            # Create sub-communities
            for cluster_id in range(n_clusters):
                cluster_members = [valid_members[i] for i in range(len(valid_members)) if labels[i] == cluster_id]
                
                if len(cluster_members) >= self.min_community_size:
                    comm_id = f"L1_C{comm_idx}"
                    
                    # Compute centroid
                    cluster_embeddings = embeddings[labels == cluster_id]
                    centroid = np.mean(cluster_embeddings, axis=0)
                    
                    # Analyze visual characteristics
                    visual_desc = self._analyze_visual_characteristics(
                        cluster_members, node_metadata
                    )
                    
                    communities[comm_id] = Community(
                        id=comm_id,
                        level=1,
                        parent_id=parent_id,
                        member_nodes=cluster_members,
                        disease_type=parent_comm.disease_type,
                        description=f"Visual cluster in {parent_comm.disease_type}: {visual_desc}",
                        centroid_embedding=centroid,
                        metadata={
                            'disease': parent_comm.disease_type,
                            'visual_features': visual_desc,
                            'size': len(cluster_members)
                        }
                    )
                    comm_idx += 1
        
        return communities
    
    def _cluster_fine_grained(self,
                             graph: nx.Graph,
                             node_embeddings: Dict[str, np.ndarray],
                             node_metadata: Dict[str, Dict],
                             parent_communities: Dict[str, Community]) -> Dict[str, Community]:
        """Level 2: Fine-grained clustering (severity/stage based)"""
        communities = {}
        comm_idx = 0
        
        for parent_id, parent_comm in parent_communities.items():
            members = parent_comm.member_nodes
            
            if len(members) < self.min_community_size * 2:
                continue
            
            # Group by class label (severity/stage)
            class_groups = defaultdict(list)
            for node in members:
                metadata = node_metadata.get(node, {})
                class_label = metadata.get('class_label', 'unknown')
                class_groups[class_label].append(node)
            
            # Create fine-grained communities
            for class_label, class_members in class_groups.items():
                if len(class_members) >= self.min_community_size:
                    comm_id = f"L2_C{comm_idx}"
                    
                    # Compute centroid
                    embeddings = [node_embeddings[node] for node in class_members if node in node_embeddings]
                    centroid = np.mean(embeddings, axis=0) if embeddings else np.zeros(512)
                    
                    communities[comm_id] = Community(
                        id=comm_id,
                        level=2,
                        parent_id=parent_id,
                        member_nodes=class_members,
                        disease_type=parent_comm.disease_type,
                        description=f"Fine-grained: {parent_comm.disease_type} - {class_label}",
                        centroid_embedding=centroid,
                        metadata={
                            'disease': parent_comm.disease_type,
                            'class_label': class_label,
                            'size': len(class_members)
                        }
                    )
                    comm_idx += 1
        
        return communities
    
    def _analyze_visual_characteristics(self,
                                       members: List[str],
                                       node_metadata: Dict[str, Dict]) -> str:
        """Analyze visual characteristics of a cluster"""
        class_labels = []
        for node in members:
            metadata = node_metadata.get(node, {})
            class_label = metadata.get('class_label', 'unknown')
            class_labels.append(class_label)
        
        # Find most common class
        if class_labels:
            from collections import Counter
            most_common = Counter(class_labels).most_common(1)[0][0]
            return f"predominantly {most_common}"
        
        return "mixed characteristics"
    
    def _build_hierarchy(self):
        """Build hierarchical relationship structure"""
        self.hierarchy = {
            'level_0': {},
            'level_1': {},
            'level_2': {}
        }
        
        for comm_id, community in self.communities.items():
            level_key = f'level_{community.level}'
            if level_key in self.hierarchy:
                self.hierarchy[level_key][comm_id] = {
                    'parent': community.parent_id,
                    'children': []
                }
        
        # Link children to parents
        for comm_id, community in self.communities.items():
            if community.parent_id:
                parent_level = f'level_{community.level - 1}'
                if parent_level in self.hierarchy and community.parent_id in self.hierarchy[parent_level]:
                    self.hierarchy[parent_level][community.parent_id]['children'].append(comm_id)
    
    def get_community_path(self, node_id: str) -> List[Community]:
        """Get hierarchical path from global to local community for a node"""
        path = []
        
        # Find node's communities at each level
        for level in [2, 1, 0]:  # Bottom-up
            for comm_id, community in self.communities.items():
                if community.level == level and node_id in community.member_nodes:
                    path.insert(0, community)
                    break
        
        return path
    
    def get_community_neighbors(self, comm_id: str, max_distance: int = 2) -> List[str]:
        """Get neighboring communities within max distance"""
        if comm_id not in self.communities:
            return []
        
        community = self.communities[comm_id]
        neighbors = set()
        
        # Add siblings (same level, same parent)
        if community.parent_id:
            for other_id, other_comm in self.communities.items():
                if (other_comm.level == community.level and 
                    other_comm.parent_id == community.parent_id and
                    other_id != comm_id):
                    neighbors.add(other_id)
        
        # Add children
        level_key = f'level_{community.level}'
        if level_key in self.hierarchy and comm_id in self.hierarchy[level_key]:
            neighbors.update(self.hierarchy[level_key][comm_id]['children'])
        
        # Add parent
        if community.parent_id:
            neighbors.add(community.parent_id)
        
        return list(neighbors)
    
    def export_communities(self) -> Dict:
        """Export community structure for serialization"""
        export_data = {
            'communities': {},
            'hierarchy': self.hierarchy,
            'stats': {
                'total_communities': len(self.communities),
                'by_level': {0: 0, 1: 0, 2: 0}
            }
        }
        
        for comm_id, community in self.communities.items():
            export_data['communities'][comm_id] = {
                'id': community.id,
                'level': community.level,
                'parent_id': community.parent_id,
                'member_count': len(community.member_nodes),
                'disease_type': community.disease_type,
                'description': community.description,
                'metadata': community.metadata
            }
            export_data['stats']['by_level'][community.level] += 1
        
        return export_data
    
    def print_summary(self):
        """Print community detection summary"""
        print("\n" + "=" * 60)
        print("COMMUNITY DETECTION SUMMARY")
        print("=" * 60)
        
        by_level = {0: [], 1: [], 2: []}
        for comm_id, community in self.communities.items():
            by_level[community.level].append(community)
        
        for level in [0, 1, 2]:
            comms = by_level[level]
            if comms:
                print(f"\nLevel {level} ({len(comms)} communities):")
                for comm in comms[:5]:  # Show first 5
                    print(f"  {comm.id}: {comm.description}")
                    print(f"    Members: {len(comm.member_nodes)}, Disease: {comm.disease_type}")
                if len(comms) > 5:
                    print(f"  ... and {len(comms) - 5} more")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Community Detection Module for Medical GraphRAG")
    print("Implements hierarchical clustering based on:")
    print("  - Disease type (Level 0)")
    print("  - Visual features (Level 1)")
    print("  - Fine-grained characteristics (Level 2)")

