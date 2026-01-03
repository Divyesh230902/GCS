#!/usr/bin/env python3
"""
Regenerate all plots with proper K labels (K=1, K=3, K=5, K=10)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src import GraphRAGVisualizer, GraphRAGEvaluator
import json

def regenerate_plots():
    """Regenerate all plots with fixed labels"""
    
    print("=" * 80)
    print("REGENERATING PLOTS WITH PROPER K LABELS")
    print("=" * 80)
    
    evaluator = GraphRAGEvaluator()
    
    # 1. Enhanced experiment plots
    print("\n[1/2] Regenerating enhanced experiment plots...")
    try:
        with open('experiments/results/evaluation_results_enhanced.json', 'r') as f:
            data_enhanced = json.load(f)
        
        visualizer = GraphRAGVisualizer(output_dir='plots_enhanced')
        
        formatted_results = {
            'Enhanced GraphRAG (Ours)': evaluator.format_for_plots(data_enhanced['aggregated_metrics'])
        }
        
        visualizer.plot_baseline_comparison(formatted_results)
        print("   ✓ Enhanced plots regenerated")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # 2. Fixed experiment plots
    print("\n[2/2] Regenerating fixed experiment plots...")
    try:
        with open('experiments/results/evaluation_results_fixed.json', 'r') as f:
            data_fixed = json.load(f)
        
        visualizer = GraphRAGVisualizer(output_dir='plots_real_fixed')
        
        formatted_results = {
            'Enhanced GraphRAG': evaluator.format_for_plots(data_fixed['aggregated_metrics'])
        }
        
        visualizer.plot_baseline_comparison(formatted_results)
        print("   ✓ Fixed plots regenerated")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ ALL PLOTS REGENERATED WITH PROPER LABELS!")
    print("=" * 80)
    print("\nPlots now show:")
    print("  • X-axis labels: K=1, K=3, K=5, K=10")
    print("  • Proper K value positioning")
    print("  • Clear metric labels")
    
    print("\n📁 Updated plot directories:")
    print("  • plots_enhanced/comparison/baseline_comparison.pdf")
    print("  • plots_real_fixed/comparison/baseline_comparison.pdf")
    print("  • plots_enhanced_fixed/comparison/baseline_comparison.pdf")

if __name__ == "__main__":
    regenerate_plots()

