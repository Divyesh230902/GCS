#!/usr/bin/env python3
"""
Clean up project directory - Remove old/duplicate files
Keep only the latest, most relevant files
"""

from pathlib import Path
import shutil
import os

def cleanup_project():
    """Clean up project directory"""
    
    print("=" * 80)
    print("PROJECT CLEANUP - REMOVING OLD/DUPLICATE FILES")
    print("=" * 80)
    
    project_root = Path.cwd()
    
    # Files/directories to remove
    to_remove = [
        # Old plot directories (keeping only latest)
        "plots_real",
        "plots_real_fixed", 
        "plots_enhanced_fixed",
        
        # Old/duplicate experiment scripts
        "run_experiments.py",  # Keep run_experiments_enhanced.py
        "run_experiments_fixed.py",  # Superseded by enhanced
        "run_ablation_study.py",  # Keep run_minimal_ablation.py
        "run_simple_ablation.py",  # Keep run_minimal_ablation.py
        
        # Old demo files
        "demo_enhanced_graphrag.py",  # Not needed anymore
        
        # Cache directories (can be regenerated)
        "embeddings_cache",
        "__pycache__",
        "src/__pycache__",
        "tests/__pycache__",
        
        # Old documentation that's been superseded
        "TROUBLESHOOTING.md",  # Info is in other docs
        "QUICK_FIX.md",  # Info is in other docs
        "SUCCESS_SUMMARY.md",  # Superseded by FINAL_SUMMARY
        "CHIIR26_PAPER_READY.md",  # Superseded by FINAL_SUMMARY
        "VISUALIZATION_SUMMARY.md",  # Info in other docs
        "QUICK_REFERENCE.md",  # Superseded
        "PROJECT_STRUCTURE.md",  # Outdated
        "IMPLEMENTATION_SUMMARY.md",  # Superseded
        
        # Temporary/test files
        "test_output",
        "temp",
        ".pytest_cache",
        
        # Old baseline file (empty)
        "baseline.py",
    ]
    
    print("\n🗑️  Removing old/duplicate files...\n")
    
    removed_count = 0
    skipped_count = 0
    
    for item in to_remove:
        item_path = project_root / item
        
        if item_path.exists():
            try:
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                    print(f"  ✓ Removed directory: {item}/")
                else:
                    item_path.unlink()
                    print(f"  ✓ Removed file: {item}")
                removed_count += 1
            except Exception as e:
                print(f"  ⚠️  Could not remove {item}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1
    
    print(f"\n📊 Cleanup summary:")
    print(f"  • Removed: {removed_count} items")
    print(f"  • Skipped (not found): {skipped_count} items")
    
    # Create a clean directory structure summary
    print("\n" + "=" * 80)
    print("CLEAN PROJECT STRUCTURE")
    print("=" * 80)
    
    print("""
📁 GCS/
├── 📂 src/                          # Source code
│   ├── clip_embeddings.py          # CLIP model
│   ├── graphRAG.py                 # GraphRAG system
│   ├── ssm.py                      # SSM query processor
│   ├── enhanced_graphrag.py        # Enhanced retriever
│   ├── community_detection.py      # Community detection
│   ├── community_summarization.py  # Summaries
│   ├── evaluation.py               # Metrics & evaluation
│   ├── visualization.py            # Plotting
│   ├── data_utils.py               # Data loading
│   ├── model_config.py             # Model configs
│   └── __init__.py
│
├── 📂 tests/                        # Test files
│   └── test_enhanced_graphrag.py
│
├── 📂 scripts/                      # Utility scripts
│   └── balanced_sampling.py        # Data balancing
│
├── 📂 data/                         # Raw datasets
│   ├── AlzheimerDataset/
│   ├── brain-tumor-mri-dataset/
│   ├── ms_slices_central/
│   └── parkinsons_dataset_processed/
│
├── 📂 balanced_data/                # Balanced datasets
│   ├── balanced_alzheimer/
│   ├── balanced_brain_tumor/
│   ├── balanced_parkinson/
│   ├── balanced_ms/
│   └── balanced_data_utils.py
│
├── 📂 experiments/                  # Experiment results
│   └── results/
│       ├── evaluation_results_enhanced.json  ✅ Main results
│       └── ablation_results.json            ✅ Ablation study
│
├── 📂 plots_enhanced/               # Main visualizations
│   ├── comparison/
│   │   ├── baseline_comparison.pdf  ✅
│   │   └── query_time.pdf
│   ├── graphrag/
│   │   └── hierarchical_graph.pdf
│   └── community/
│       └── community_stats.pdf
│
├── 📂 plots_by_disease/             # Disease-specific plots
│   ├── performance_by_disease_bars.pdf     ✅ NEW!
│   ├── performance_by_disease_grouped.pdf  ✅ NEW!
│   ├── performance_by_disease_heatmap.pdf  ✅ NEW!
│   └── query_distribution_by_disease.pdf   ✅ NEW!
│
├── 📂 docs/                         # Documentation
│   ├── README.md
│   └── GRAPHRAG_APPROACH.md
│
├── 🐍 Main Scripts:
│   ├── run_experiments_enhanced.py  ✅ Main experiment (21 queries)
│   ├── run_minimal_ablation.py      ✅ Ablation study
│   ├── plot_by_disease.py           ✅ Disease plots
│   ├── regenerate_plots.py          ✅ Regenerate all plots
│   ├── demo_plots.py                   Synthetic demo
│   └── main.py                         Entry point
│
├── 📄 Key Documentation:
│   ├── FINAL_SUMMARY_CHIIR26.md     ✅ Complete project summary
│   ├── ENHANCED_RESULTS_SUMMARY.md  ✅ Main results analysis
│   ├── ABLATION_STUDY_RESULTS.md    ✅ Ablation analysis
│   ├── EVALUATION_GUIDE.md          ✅ Methodology
│   ├── PLOT_GUIDE.md                   Plot reference
│   └── CHIIR26_SUMMARY.md              Paper outline
│
├── ⚙️ Configuration:
│   ├── requirements.txt             # Dependencies
│   ├── setup.py                     # Package setup
│   ├── .gitignore
│   └── ReadMe                       # Original readme
│
└── 🧪 Utilities:
    ├── run_tests.py                 # Test runner
    └── cleanup_project.py           # This script
    """)
    
    print("\n" + "=" * 80)
    print("✅ PROJECT CLEANED UP!")
    print("=" * 80)
    
    print("\n📌 Key Files to Use:")
    print("  • Run experiments: python run_experiments_enhanced.py")
    print("  • Ablation study: python run_minimal_ablation.py")
    print("  • Disease plots: python plot_by_disease.py")
    print("  • Regenerate plots: python regenerate_plots.py")
    
    print("\n📊 Results & Plots:")
    print("  • Main results: experiments/results/evaluation_results_enhanced.json")
    print("  • Ablation: experiments/results/ablation_results.json")
    print("  • Main plots: plots_enhanced/")
    print("  • Disease plots: plots_by_disease/")
    
    print("\n📚 Documentation:")
    print("  • Project summary: FINAL_SUMMARY_CHIIR26.md")
    print("  • Results analysis: ENHANCED_RESULTS_SUMMARY.md")
    print("  • Ablation analysis: ABLATION_STUDY_RESULTS.md")
    
    # Count remaining files
    print("\n📈 Project Statistics:")
    
    py_files = list(project_root.glob("*.py"))
    md_files = list(project_root.glob("*.md"))
    src_files = list((project_root / "src").glob("*.py")) if (project_root / "src").exists() else []
    
    print(f"  • Python scripts (root): {len(py_files)}")
    print(f"  • Source files (src/): {len(src_files)}")
    print(f"  • Documentation files: {len(md_files)}")
    
    result_files = list((project_root / "experiments/results").glob("*.json")) if (project_root / "experiments/results").exists() else []
    print(f"  • Result files: {len(result_files)}")
    
    plot_dirs = ["plots_enhanced", "plots_by_disease"]
    total_plots = 0
    for plot_dir in plot_dirs:
        if (project_root / plot_dir).exists():
            plots = list((project_root / plot_dir).rglob("*.pdf"))
            total_plots += len(plots)
            print(f"  • Plots in {plot_dir}/: {len(plots)}")
    
    print(f"  • Total plots: {total_plots}")
    
    return removed_count

if __name__ == "__main__":
    try:
        removed = cleanup_project()
        print(f"\n🎉 Successfully cleaned up project! ({removed} items removed)")
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()

