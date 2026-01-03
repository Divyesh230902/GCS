#!/usr/bin/env python3
"""
Create plots showing performance breakdown by disease/anomaly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

def extract_disease_from_query_id(query_id):
    """Extract disease name from query ID"""
    if 'ALZ' in query_id:
        return "Alzheimer's"
    elif 'TUM' in query_id:
        return "Brain Tumor"
    elif 'PARK' in query_id:
        return "Parkinson's"
    elif 'MS' in query_id and 'CROSS' not in query_id:
        return "Multiple Sclerosis"
    elif 'CROSS' in query_id:
        return "Cross-Dataset"
    else:
        return "Other"

def plot_performance_by_disease():
    """Generate plots showing performance breakdown by disease"""
    
    print("=" * 80)
    print("GENERATING PERFORMANCE BY DISEASE/ANOMALY PLOTS")
    print("=" * 80)
    
    # Load results
    print("\nLoading results...")
    with open('experiments/results/evaluation_results_enhanced.json', 'r') as f:
        data = json.load(f)
    
    # Group queries by disease
    disease_metrics = defaultdict(lambda: {
        'precision@5': [],
        'ndcg@5': [],
        'mrr': [],
        'queries': []
    })
    
    for query_id, metrics in data['per_query_results'].items():
        disease = extract_disease_from_query_id(query_id)
        disease_metrics[disease]['precision@5'].append(metrics['precision@k']['5'])
        disease_metrics[disease]['ndcg@5'].append(metrics['ndcg@k']['5'])
        disease_metrics[disease]['mrr'].append(metrics['mrr'])
        disease_metrics[disease]['queries'].append(query_id)
    
    # Calculate averages
    disease_avg = {}
    for disease, metrics in disease_metrics.items():
        disease_avg[disease] = {
            'P@5': np.mean(metrics['precision@5']),
            'NDCG@5': np.mean(metrics['ndcg@5']),
            'MRR': np.mean(metrics['mrr']),
            'num_queries': len(metrics['queries'])
        }
    
    print("\n📊 Performance by Disease/Anomaly:")
    for disease, metrics in sorted(disease_avg.items()):
        print(f"\n{disease}:")
        print(f"  P@5: {metrics['P@5']:.3f} ({metrics['P@5']*100:.1f}%)")
        print(f"  NDCG@5: {metrics['NDCG@5']:.3f}")
        print(f"  MRR: {metrics['MRR']:.3f}")
        print(f"  Queries: {metrics['num_queries']}")
    
    # Create plots
    output_dir = Path('plots_by_disease')
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("Set2")
    
    # Plot 1: Bar chart comparison
    print("\n[1/3] Creating bar chart comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    diseases = sorted(disease_avg.keys())
    colors = sns.color_palette("Set2", len(diseases))
    
    metrics_to_plot = ['P@5', 'NDCG@5', 'MRR']
    metric_names = ['Precision@5', 'NDCG@5', 'Mean Reciprocal Rank']
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]
        values = [disease_avg[d][metric] for d in diseases]
        
        bars = ax.bar(range(len(diseases)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(name, fontsize=13, fontweight='bold')
        ax.set_title(f'{name} by Disease', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(diseases)))
        ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.0)
        
        # Highlight best performer
        best_idx = values.index(max(values))
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    save_path = output_dir / 'performance_by_disease_bars.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()
    
    # Plot 2: Grouped bar chart (all metrics together)
    print("\n[2/3] Creating grouped comparison...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(diseases))
    width = 0.25
    
    for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        values = [disease_avg[d][metric] for d in diseases]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=name, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison Across Diseases/Anomalies', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    save_path = output_dir / 'performance_by_disease_grouped.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()
    
    # Plot 3: Heatmap
    print("\n[3/3] Creating heatmap...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap
    heatmap_data = []
    for disease in diseases:
        row = [
            disease_avg[disease]['P@5'],
            disease_avg[disease]['NDCG@5'],
            disease_avg[disease]['MRR']
        ]
        heatmap_data.append(row)
    
    heatmap_data = np.array(heatmap_data)
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(diseases)))
    ax.set_xticklabels(metric_names, fontsize=12, fontweight='bold')
    ax.set_yticklabels(diseases, fontsize=12, fontweight='bold')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(diseases)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax.set_title("Performance Heatmap by Disease/Anomaly", fontsize=15, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / 'performance_by_disease_heatmap.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()
    
    # Plot 4: Number of queries per disease
    print("\n[4/4] Creating query distribution plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    query_counts = [disease_avg[d]['num_queries'] for d in diseases]
    bars = ax.bar(range(len(diseases)), query_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, query_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Queries', fontsize=13, fontweight='bold')
    ax.set_title('Query Distribution by Disease/Anomaly', fontsize=15, fontweight='bold')
    ax.set_xticks(range(len(diseases)))
    ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_path = output_dir / 'query_distribution_by_disease.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {save_path}")
    plt.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ ALL DISEASE-SPECIFIC PLOTS GENERATED!")
    print("=" * 80)
    
    print("\n📁 Generated files:")
    print(f"  • {output_dir}/performance_by_disease_bars.pdf")
    print(f"  • {output_dir}/performance_by_disease_grouped.pdf")
    print(f"  • {output_dir}/performance_by_disease_heatmap.pdf")
    print(f"  • {output_dir}/query_distribution_by_disease.pdf")
    
    print("\n📊 Key Findings:")
    best_p5 = max(disease_avg.items(), key=lambda x: x[1]['P@5'])
    best_mrr = max(disease_avg.items(), key=lambda x: x[1]['MRR'])
    
    print(f"  • Best P@5: {best_p5[0]} ({best_p5[1]['P@5']:.3f})")
    print(f"  • Best MRR: {best_mrr[0]} ({best_mrr[1]['MRR']:.3f})")
    print(f"  • Total queries: {sum(m['num_queries'] for m in disease_avg.values())}")
    print(f"  • Diseases tested: {len(diseases)}")
    
    return disease_avg

if __name__ == "__main__":
    results = plot_performance_by_disease()
    print("\n🎉 Success! Your plots now show performance by disease/anomaly!")

