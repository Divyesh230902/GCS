#!/usr/bin/env python3
"""
Demo: Generate Example Plots for CHIIR'26 Paper
Creates synthetic data and generates all plot types
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.visualization import GraphRAGVisualizer
import numpy as np
import networkx as nx
from dataclasses import dataclass

@dataclass
class MockCommunity:
    """Mock community for demonstration"""
    id: str
    level: int
    parent_id: str
    member_nodes: list
    disease_type: str
    centroid_embedding: np.ndarray

def generate_synthetic_results():
    """Generate synthetic experimental results"""
    
    # 1. Ablation Study: Component removal
    ablation_components = {
        'Full System': {
            'precision': 0.85,
            'recall': 0.82,
            'ndcg': 0.88,
            'f1': 0.835
        },
        'No Communities': {
            'precision': 0.72,
            'recall': 0.70,
            'ndcg': 0.75,
            'f1': 0.71
        },
        'No SSM': {
            'precision': 0.78,
            'recall': 0.76,
            'ndcg': 0.80,
            'f1': 0.77
        },
        'No Hierarchy': {
            'precision': 0.76,
            'recall': 0.74,
            'ndcg': 0.78,
            'f1': 0.75
        },
        'CLIP Only': {
            'precision': 0.68,
            'recall': 0.65,
            'ndcg': 0.70,
            'f1': 0.665
        }
    }
    
    # 2. Ablation Study: Hierarchy levels
    ablation_hierarchy = {
        1: {'precision': 0.70, 'recall': 0.68, 'ndcg': 0.72, 'f1': 0.69},
        2: {'precision': 0.78, 'recall': 0.76, 'ndcg': 0.80, 'f1': 0.77},
        3: {'precision': 0.85, 'recall': 0.82, 'ndcg': 0.88, 'f1': 0.835},
        4: {'precision': 0.84, 'recall': 0.81, 'ndcg': 0.87, 'f1': 0.825}
    }
    
    # 3. Ablation Study: Search modes
    ablation_search = {
        'Global': {'precision': 0.78, 'recall': 0.88, 'ndcg': 0.82, 'f1': 0.83},
        'Local': {'precision': 0.90, 'recall': 0.76, 'ndcg': 0.85, 'f1': 0.82},
        'Hybrid': {'precision': 0.85, 'recall': 0.82, 'ndcg': 0.88, 'f1': 0.835},
        'Auto': {'precision': 0.84, 'recall': 0.83, 'ndcg': 0.87, 'f1': 0.835}
    }
    
    # 4. Ablation Study: Feature weighting
    ablation_weighting = {
        (1.0, 0.0): {'precision': 0.75, 'recall': 0.73, 'ndcg': 0.77},
        (0.8, 0.2): {'precision': 0.80, 'recall': 0.78, 'ndcg': 0.82},
        (0.6, 0.4): {'precision': 0.85, 'recall': 0.82, 'ndcg': 0.88},
        (0.5, 0.5): {'precision': 0.83, 'recall': 0.81, 'ndcg': 0.86},
        (0.4, 0.6): {'precision': 0.81, 'recall': 0.79, 'ndcg': 0.84},
        (0.2, 0.8): {'precision': 0.78, 'recall': 0.76, 'ndcg': 0.80},
        (0.0, 1.0): {'precision': 0.72, 'recall': 0.70, 'ndcg': 0.75}
    }
    
    # 5. Baseline Comparison
    k_values = list(range(1, 11))
    baseline_comparison = {
        'Enhanced GraphRAG (Ours)': {
            'precision@k': [0.92, 0.90, 0.88, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80],
            'recall@k': [0.18, 0.36, 0.52, 0.65, 0.75, 0.82, 0.87, 0.90, 0.92, 0.94],
            'ndcg@k': [0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83],
            'map': [0.88] * 10
        },
        'FAISS Vector Search': {
            'precision@k': [0.82, 0.78, 0.75, 0.72, 0.70, 0.68, 0.66, 0.65, 0.64, 0.63],
            'recall@k': [0.16, 0.31, 0.45, 0.56, 0.65, 0.72, 0.77, 0.81, 0.84, 0.86],
            'ndcg@k': [0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.69, 0.68, 0.67],
            'map': [0.75] * 10
        },
        'Basic RAG': {
            'precision@k': [0.78, 0.75, 0.72, 0.70, 0.68, 0.66, 0.65, 0.64, 0.63, 0.62],
            'recall@k': [0.15, 0.30, 0.43, 0.54, 0.63, 0.70, 0.75, 0.79, 0.82, 0.84],
            'ndcg@k': [0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.67, 0.66, 0.65, 0.64],
            'map': [0.72] * 10
        },
        'CLIP-only': {
            'precision@k': [0.75, 0.72, 0.69, 0.67, 0.65, 0.63, 0.62, 0.61, 0.60, 0.59],
            'recall@k': [0.15, 0.29, 0.41, 0.52, 0.60, 0.67, 0.72, 0.76, 0.79, 0.81],
            'ndcg@k': [0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.64, 0.63, 0.62, 0.61],
            'map': [0.68] * 10
        }
    }
    
    # 6. Query Time Comparison
    query_time = {
        'Enhanced GraphRAG': {'latency': 45.2, 'throughput': 22.1},
        'FAISS': {'latency': 12.5, 'throughput': 80.0},
        'Basic RAG': {'latency': 35.8, 'throughput': 27.9},
        'CLIP-only': {'latency': 8.3, 'throughput': 120.5}
    }
    
    return {
        'ablation_components': ablation_components,
        'ablation_hierarchy': ablation_hierarchy,
        'ablation_search': ablation_search,
        'ablation_weighting': ablation_weighting,
        'baseline_comparison': baseline_comparison,
        'query_time': query_time
    }

def create_mock_communities():
    """Create mock communities for visualization"""
    communities = {}
    
    # Level 0: Disease categories
    diseases = ['alzheimer', 'brain_tumor', 'parkinson']
    for i, disease in enumerate(diseases):
        comm_id = f"L0_C{i}_{disease}"
        communities[comm_id] = MockCommunity(
            id=comm_id,
            level=0,
            parent_id=None,
            member_nodes=[f"{disease}_img_{j}" for j in range(10)],
            disease_type=disease,
            centroid_embedding=np.random.randn(512).astype(np.float32)
        )
    
    # Level 1: Visual feature groups
    for i in range(6):
        parent_disease = diseases[i % 3]
        comm_id = f"L1_C{i}"
        parent_id = f"L0_C{i % 3}_{parent_disease}"
        communities[comm_id] = MockCommunity(
            id=comm_id,
            level=1,
            parent_id=parent_id,
            member_nodes=[f"{parent_disease}_img_{j}" for j in range(i*2, i*2+5)],
            disease_type=parent_disease,
            centroid_embedding=np.random.randn(512).astype(np.float32)
        )
    
    # Level 2: Fine-grained
    for i in range(9):
        parent_disease = diseases[i % 3]
        comm_id = f"L2_C{i}"
        parent_id = f"L1_C{i % 6}"
        communities[comm_id] = MockCommunity(
            id=comm_id,
            level=2,
            parent_id=parent_id,
            member_nodes=[f"{parent_disease}_img_{j}" for j in range(i, i+3)],
            disease_type=parent_disease,
            centroid_embedding=np.random.randn(512).astype(np.float32)
        )
    
    return communities

def create_mock_graph(communities):
    """Create mock graph from communities"""
    G = nx.Graph()
    
    # Add all nodes from communities
    all_nodes = set()
    for comm in communities.values():
        all_nodes.update(comm.member_nodes)
    
    for node in all_nodes:
        G.add_node(node)
    
    # Add some random edges
    nodes_list = list(all_nodes)
    for _ in range(len(nodes_list) * 2):
        n1, n2 = np.random.choice(nodes_list, 2, replace=False)
        G.add_edge(n1, n2, weight=np.random.rand())
    
    return G

def main():
    print("=" * 70)
    print("GENERATING DEMO PLOTS FOR CHIIR'26 PAPER")
    print("=" * 70)
    
    # Initialize visualizer
    visualizer = GraphRAGVisualizer(output_dir="plots")
    
    # Generate synthetic results
    print("\n1. Generating synthetic experimental results...")
    results = generate_synthetic_results()
    print("   ✓ Results generated")
    
    # Generate mock communities and graph
    print("\n2. Creating mock communities and graph...")
    communities = create_mock_communities()
    graph = create_mock_graph(communities)
    print(f"   ✓ Created {len(communities)} communities, {graph.number_of_nodes()} nodes")
    
    # Generate plots
    print("\n3. Generating ablation study plots...")
    
    print("\n   a) Component ablation...")
    visualizer.plot_ablation_components(
        results['ablation_components'],
        metrics=['precision', 'recall', 'ndcg']
    )
    
    print("\n   b) Hierarchy level ablation...")
    visualizer.plot_ablation_hierarchy_levels(
        results['ablation_hierarchy']
    )
    
    print("\n   c) Search mode ablation...")
    visualizer.plot_ablation_search_modes(
        results['ablation_search']
    )
    
    print("\n   d) Feature weighting ablation...")
    visualizer.plot_ablation_weighting(
        results['ablation_weighting']
    )
    
    print("\n4. Generating GraphRAG-specific plots...")
    
    print("\n   a) Hierarchical graph structure...")
    visualizer.plot_hierarchical_graph(graph, communities)
    
    print("\n   b) Community statistics...")
    visualizer.plot_community_statistics(communities)
    
    print("\n   c) Community similarity matrix...")
    visualizer.plot_community_similarity_matrix(communities)
    
    print("\n5. Generating comparison plots...")
    
    print("\n   a) Baseline comparison...")
    visualizer.plot_baseline_comparison(
        results['baseline_comparison']
    )
    
    print("\n   b) Query time comparison...")
    visualizer.plot_query_time_comparison(
        results['query_time']
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nPlots saved to: {visualizer.output_dir}")
    print("\nPlot Categories:")
    print("  📊 plots/ablation/        - Ablation study plots")
    print("  📊 plots/graphrag/        - GraphRAG structure plots")
    print("  📊 plots/community/       - Community analysis plots")
    print("  📊 plots/comparison/      - Baseline comparison plots")
    
    print("\nPlots for CHIIR'26 Paper:")
    print("  ✓ ablation_components.pdf     - Component removal ablation")
    print("  ✓ ablation_hierarchy.pdf      - Hierarchy depth ablation")
    print("  ✓ ablation_search_modes.pdf   - Search strategy ablation")
    print("  ✓ ablation_weighting.pdf      - Feature weighting ablation")
    print("  ✓ hierarchical_graph.pdf      - 3-level graph structure")
    print("  ✓ community_stats.pdf         - Community statistics")
    print("  ✓ community_similarity.pdf    - Community similarity matrix")
    print("  ✓ baseline_comparison.pdf     - Performance comparison")
    print("  ✓ query_time.pdf              - Latency & throughput")
    
    print("\n" + "=" * 70)
    print("📝 These plots are ready for your CHIIR'26 paper!")
    print("   You can customize them with real experimental data.")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()

