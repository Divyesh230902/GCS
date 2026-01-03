"""
Visualization Module for CHIIR'26 Paper
Generates plots for ablation studies, GraphRAG analysis, and paper figures
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class GraphRAGVisualizer:
    """
    Comprehensive visualization suite for GraphRAG paper
    """
    
    def __init__(self, output_dir: str = "plots"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "ablation").mkdir(exist_ok=True)
        (self.output_dir / "graphrag").mkdir(exist_ok=True)
        (self.output_dir / "comparison").mkdir(exist_ok=True)
        (self.output_dir / "community").mkdir(exist_ok=True)
        
        print(f"📊 Visualizer initialized. Plots will be saved to: {self.output_dir}")
    
    # ==================== ABLATION STUDY PLOTS ====================
    
    def plot_ablation_components(self, 
                                 results: Dict[str, Dict[str, float]],
                                 metrics: List[str] = ['precision', 'recall', 'ndcg'],
                                 save_name: str = "ablation_components.pdf"):
        """
        Plot ablation study: Impact of removing components
        
        Args:
            results: Dict of {config_name: {metric: value}}
                    e.g., {'Full System': {'precision': 0.85, ...}, 
                           'No Communities': {'precision': 0.72, ...}}
            metrics: List of metrics to plot
            save_name: Output filename
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        configs = list(results.keys())
        colors = sns.color_palette("husl", len(configs))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [results[config][metric] for config in configs]
            
            bars = ax.bar(range(len(configs)), values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        save_path = self.output_dir / "ablation" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_ablation_hierarchy_levels(self,
                                       results: Dict[int, Dict[str, float]],
                                       save_name: str = "ablation_hierarchy.pdf"):
        """
        Plot ablation: Impact of different hierarchy levels
        
        Args:
            results: Dict of {num_levels: {metric: value}}
                    e.g., {1: {'precision': 0.70, ...}, 2: {...}, 3: {...}}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        levels = sorted(results.keys())
        metrics = list(results[levels[0]].keys())
        
        x = np.arange(len(levels))
        width = 0.2
        colors = sns.color_palette("Set2", len(metrics))
        
        for i, metric in enumerate(metrics):
            values = [results[level][metric] for level in levels]
            offset = (i - len(metrics)/2) * width + width/2
            bars = ax.bar(x + offset, values, width, label=metric.capitalize(), 
                         color=colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Number of Hierarchy Levels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Impact of Hierarchy Depth on Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l} Level{"s" if l>1 else ""}' for l in levels])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        save_path = self.output_dir / "ablation" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_ablation_search_modes(self,
                                   results: Dict[str, Dict[str, float]],
                                   save_name: str = "ablation_search_modes.pdf"):
        """
        Plot ablation: Performance of different search modes
        
        Args:
            results: Dict of {mode: {metric: value}}
                    e.g., {'Global': {...}, 'Local': {...}, 'Hybrid': {...}}
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        modes = list(results.keys())
        metrics = list(results[modes[0]].keys())
        
        # Plot 1: Grouped bar chart
        x = np.arange(len(modes))
        width = 0.25
        colors = sns.color_palette("muted", len(metrics))
        
        for i, metric in enumerate(metrics):
            values = [results[mode][metric] for mode in modes]
            offset = (i - len(metrics)/2) * width + width/2
            bars = ax1.bar(x + offset, values, width, label=metric.capitalize(),
                          color=colors[i], alpha=0.8, edgecolor='black')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Search Mode', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Performance by Search Mode', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modes)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Plot 2: Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        
        for mode, color in zip(modes, sns.color_palette("husl", len(modes))):
            values = [results[mode][metric] for metric in metrics]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=mode, color=color)
            ax2.fill(angles, values, alpha=0.15, color=color)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([m.capitalize() for m in metrics], fontsize=10)
        ax2.set_ylim(0, 1.0)
        ax2.set_title('Search Mode Comparison\n(Radar View)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = self.output_dir / "ablation" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_ablation_weighting(self,
                               results: Dict[Tuple[float, float], Dict[str, float]],
                               save_name: str = "ablation_weighting.pdf"):
        """
        Plot ablation: Impact of disease vs. visual feature weighting
        
        Args:
            results: Dict of {(disease_weight, visual_weight): {metric: value}}
                    e.g., {(0.8, 0.2): {'precision': 0.82, ...}, ...}
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        weightings = sorted(results.keys())
        disease_weights = [w[0] for w in weightings]
        
        metrics = list(results[weightings[0]].keys())
        colors = sns.color_palette("Set2", len(metrics))
        
        # Plot 1: Line plot
        ax = axes[0]
        for i, metric in enumerate(metrics):
            values = [results[w][metric] for w in weightings]
            ax.plot(disease_weights, values, 'o-', linewidth=2, 
                   markersize=8, label=metric.capitalize(), color=colors[i])
        
        ax.set_xlabel('Disease Weight (Visual Weight = 1 - Disease)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Impact of Feature Weighting', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Plot 2: Heatmap
        ax = axes[1]
        metric_to_show = metrics[0]  # Show primary metric
        
        # Create matrix for heatmap
        unique_disease = sorted(set([w[0] for w in weightings]))
        unique_visual = sorted(set([w[1] for w in weightings]))
        
        matrix = np.zeros((len(unique_visual), len(unique_disease)))
        for w, vals in results.items():
            i = unique_visual.index(w[1])
            j = unique_disease.index(w[0])
            matrix[i, j] = vals[metric_to_show]
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(unique_disease)))
        ax.set_yticks(range(len(unique_visual)))
        ax.set_xticklabels([f'{w:.1f}' for w in unique_disease])
        ax.set_yticklabels([f'{w:.1f}' for w in unique_visual])
        ax.set_xlabel('Disease Weight', fontsize=12, fontweight='bold')
        ax.set_ylabel('Visual Weight', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_to_show.capitalize()} Heatmap', 
                    fontsize=14, fontweight='bold')
        
        # Add value annotations
        for i in range(len(unique_visual)):
            for j in range(len(unique_disease)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        save_path = self.output_dir / "ablation" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    # ==================== GRAPHRAG STRUCTURE PLOTS ====================
    
    def plot_hierarchical_graph(self,
                               graph: nx.Graph,
                               communities: Dict[str, any],
                               save_name: str = "hierarchical_graph.pdf"):
        """
        Visualize hierarchical community structure
        
        Args:
            graph: NetworkX graph
            communities: Dictionary of Community objects
            save_name: Output filename
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Organize communities by level
        by_level = {0: [], 1: [], 2: []}
        for comm_id, comm in communities.items():
            by_level[comm.level].append(comm)
        
        for level, ax in enumerate(axes):
            # Create subgraph for this level
            level_communities = by_level[level]
            
            if not level_communities:
                ax.text(0.5, 0.5, 'No communities at this level',
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'Level {level}', fontsize=14, fontweight='bold')
                ax.axis('off')
                continue
            
            # Get all nodes in level communities
            level_nodes = set()
            for comm in level_communities:
                level_nodes.update(comm.member_nodes)
            
            # Create subgraph
            subgraph = graph.subgraph(level_nodes)
            
            # Layout
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
            
            # Color nodes by community
            node_colors = []
            color_map = {}
            colors = sns.color_palette("husl", len(level_communities))
            
            for i, comm in enumerate(level_communities):
                color_map[comm.id] = colors[i]
            
            for node in subgraph.nodes():
                # Find which community this node belongs to
                for comm in level_communities:
                    if node in comm.member_nodes:
                        node_colors.append(color_map[comm.id])
                        break
            
            # Draw
            nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                                  node_size=100, alpha=0.8, ax=ax)
            nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=ax)
            
            # Title
            level_names = ['Global (Disease)', 'Mid (Visual)', 'Local (Fine-grained)']
            ax.set_title(f'Level {level}: {level_names[level]}\n'
                        f'{len(level_communities)} communities, '
                        f'{len(level_nodes)} nodes',
                        fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / "graphrag" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_community_statistics(self,
                                 communities: Dict[str, any],
                                 save_name: str = "community_stats.pdf"):
        """
        Plot community statistics
        
        Args:
            communities: Dictionary of Community objects
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Organize by level
        by_level = {0: [], 1: [], 2: []}
        for comm in communities.values():
            by_level[comm.level].append(comm)
        
        # Plot 1: Community sizes by level
        ax = axes[0, 0]
        level_labels = ['L0\n(Global)', 'L1\n(Mid)', 'L2\n(Local)']
        
        for level in [0, 1, 2]:
            sizes = [len(comm.member_nodes) for comm in by_level[level]]
            if sizes:
                positions = np.random.normal(level, 0.04, size=len(sizes))
                ax.scatter(positions, sizes, alpha=0.6, s=100)
        
        ax.boxplot([([len(comm.member_nodes) for comm in by_level[level]] or [0]) 
                    for level in [0, 1, 2]],
                   labels=level_labels, showfliers=False)
        ax.set_ylabel('Community Size (# nodes)', fontsize=12, fontweight='bold')
        ax.set_title('Community Size Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Number of communities per level
        ax = axes[0, 1]
        counts = [len(by_level[level]) for level in [0, 1, 2]]
        colors = sns.color_palette("Set2", 3)
        bars = ax.bar(['Level 0', 'Level 1', 'Level 2'], counts, 
                     color=colors, alpha=0.8, edgecolor='black')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Number of Communities', fontsize=12, fontweight='bold')
        ax.set_title('Communities per Hierarchy Level', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 3: Disease distribution
        ax = axes[1, 0]
        disease_counts = {}
        for comm in communities.values():
            disease = comm.disease_type
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        diseases = list(disease_counts.keys())
        counts = list(disease_counts.values())
        colors = sns.color_palette("husl", len(diseases))
        
        wedges, texts, autotexts = ax.pie(counts, labels=diseases, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        ax.set_title('Disease Distribution in Communities', 
                    fontsize=14, fontweight='bold')
        
        # Plot 4: Hierarchy tree
        ax = axes[1, 1]
        
        # Count parent-child relationships
        level_0_comms = [c for c in communities.values() if c.level == 0]
        level_1_comms = [c for c in communities.values() if c.level == 1]
        level_2_comms = [c for c in communities.values() if c.level == 2]
        
        # Draw tree structure
        y_positions = {0: 0.8, 1: 0.5, 2: 0.2}
        
        for level in [0, 1, 2]:
            comms_at_level = [c for c in communities.values() if c.level == level]
            if comms_at_level:
                x_positions = np.linspace(0.1, 0.9, len(comms_at_level))
                for x, comm in zip(x_positions, comms_at_level):
                    circle = plt.Circle((x, y_positions[level]), 0.03, 
                                       color=sns.color_palette("husl", 3)[level],
                                       alpha=0.8)
                    ax.add_patch(circle)
                    ax.text(x, y_positions[level], str(len(comm.member_nodes)),
                           ha='center', va='center', fontsize=8, 
                           color='white', fontweight='bold')
        
        # Draw connections (simplified)
        for comm in level_1_comms:
            if comm.parent_id and comm.parent_id in communities:
                parent_idx = [c.id for c in level_0_comms].index(comm.parent_id)
                child_idx = [c.id for c in level_1_comms].index(comm.id)
                x1 = np.linspace(0.1, 0.9, len(level_0_comms))[parent_idx]
                x2 = np.linspace(0.1, 0.9, len(level_1_comms))[child_idx]
                ax.plot([x1, x2], [y_positions[0], y_positions[1]], 
                       'k-', alpha=0.2, linewidth=1)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.5, 0.8])
        ax.set_yticklabels(['Level 2\n(Local)', 'Level 1\n(Mid)', 'Level 0\n(Global)'])
        ax.set_title('Hierarchical Structure', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / "community" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_community_similarity_matrix(self,
                                        communities: Dict[str, any],
                                        save_name: str = "community_similarity.pdf"):
        """
        Plot similarity matrix between communities
        
        Args:
            communities: Dictionary of Community objects
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        comm_ids = list(communities.keys())
        n = len(comm_ids)
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((n, n))
        
        for i, id1 in enumerate(comm_ids):
            for j, id2 in enumerate(comm_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    emb1 = communities[id1].centroid_embedding
                    emb2 = communities[id2].centroid_embedding
                    
                    # Cosine similarity
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarity_matrix[i, j] = similarity
        
        # Plot
        im = ax.imshow(similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        
        # Labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([id.split('_')[0] for id in comm_ids], rotation=90, fontsize=8)
        ax.set_yticklabels([id.split('_')[0] for id in comm_ids], fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12, fontweight='bold')
        
        ax.set_title('Inter-Community Similarity Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / "community" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    # ==================== COMPARISON PLOTS ====================
    
    def plot_baseline_comparison(self,
                                results: Dict[str, Dict[str, List[float]]],
                                save_name: str = "baseline_comparison.pdf"):
        """
        Compare with baseline methods
        
        Args:
            results: Dict of {method: {metric: [values for different K]}}
                    e.g., {'Enhanced GraphRAG': {'precision@k': [0.9, 0.85, 0.82, ...]},
                           'FAISS': {'precision@k': [0.7, 0.68, 0.65, ...]}}
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        methods = list(results.keys())
        # Get the actual K values from the data (should be [1, 3, 5, 10])
        first_method = methods[0]
        if isinstance(results[first_method]['precision@k'], dict):
            k_values = sorted(results[first_method]['precision@k'].keys())
        else:
            k_values = [1, 3, 5, 10]  # Default K values
        
        colors = sns.color_palette("Set1", len(methods))
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        
        metrics = ['precision@k', 'recall@k', 'ndcg@k', 'map']
        metric_names = ['Precision@K', 'Recall@K', 'NDCG@K', 'MAP@K']
        
        for idx, (metric, name) in enumerate(zip(metrics[:3], metric_names[:3])):
            ax = axes[idx]
            
            for i, (method, color, marker) in enumerate(zip(methods, colors, markers)):
                if metric in results[method]:
                    values = results[method][metric]
                    if isinstance(values, dict):
                        # Extract values in order of k_values
                        y_values = [values[k] for k in k_values]
                    else:
                        y_values = values
                    ax.plot(k_values, y_values, marker=marker, linewidth=2.5,
                           markersize=8, label=method, color=color, alpha=0.8)
            
            ax.set_xlabel('K (Top-K Results)', fontsize=12, fontweight='bold')
            ax.set_ylabel(name, fontsize=12, fontweight='bold')
            ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.0)
            # Set proper x-axis ticks with K values
            ax.set_xticks(k_values)
            ax.set_xticklabels([f'K={k}' for k in k_values])
        
        # Plot 4: Overall comparison (bar chart)
        ax = axes[3]
        
        # Average over K for each metric
        avg_results = {}
        for method in methods:
            avg_results[method] = {
                metric: np.mean(results[method].get(metric, [0]))
                for metric in ['precision@k', 'recall@k', 'ndcg@k']
            }
        
        x = np.arange(len(metrics[:3]))
        width = 0.8 / len(methods)
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            offset = (i - len(methods)/2) * width + width/2
            values = [avg_results[method][m] for m in ['precision@k', 'recall@k', 'ndcg@k']]
            bars = ax.bar(x + offset, values, width, label=method, 
                         color=color, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Precision', 'Recall', 'NDCG'])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        save_path = self.output_dir / "comparison" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_query_time_comparison(self,
                                   results: Dict[str, Dict[str, float]],
                                   save_name: str = "query_time.pdf"):
        """
        Compare query latency across methods
        
        Args:
            results: Dict of {method: {'latency': time_in_ms, 'throughput': queries_per_sec}}
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        methods = list(results.keys())
        colors = sns.color_palette("Set2", len(methods))
        
        # Plot 1: Latency
        latencies = [results[method]['latency'] for method in methods]
        bars = ax1.barh(range(len(methods)), latencies, color=colors, 
                       alpha=0.8, edgecolor='black')
        
        for bar, lat in zip(bars, latencies):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {lat:.1f} ms',
                    ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax1.set_yticks(range(len(methods)))
        ax1.set_yticklabels(methods)
        ax1.set_xlabel('Query Latency (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Average Query Latency', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Throughput
        throughputs = [results[method]['throughput'] for method in methods]
        bars = ax2.barh(range(len(methods)), throughputs, color=colors,
                       alpha=0.8, edgecolor='black')
        
        for bar, thr in zip(bars, throughputs):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {thr:.1f} q/s',
                    ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax2.set_yticks(range(len(methods)))
        ax2.set_yticklabels(methods)
        ax2.set_xlabel('Throughput (queries/sec)', fontsize=12, fontweight='bold')
        ax2.set_title('Query Throughput', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "comparison" / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    # ==================== UTILITY METHODS ====================
    
    def create_summary_figure(self,
                             all_results: Dict,
                             save_name: str = "paper_summary.pdf"):
        """
        Create comprehensive summary figure for paper
        
        Args:
            all_results: Dictionary containing all experimental results
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # This would combine multiple plots into one summary figure
        # Implementation depends on specific results format
        
        print(f"✓ Summary figure would be created here")
        plt.close()


if __name__ == "__main__":
    print("GraphRAG Visualization Module for CHIIR'26")
    print("Generates publication-quality plots for ablation studies and analysis")

