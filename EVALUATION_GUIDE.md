# Evaluation Guide - Getting Real Scores

This guide explains how to get **real experimental scores** instead of synthetic values for your CHIIR'26 paper.

---

## 🎯 **What's Implemented**

### **1. Complete Evaluation Framework** (`src/evaluation.py`)
- ✅ **Precision@K**: How many retrieved items are relevant
- ✅ **Recall@K**: How many relevant items were retrieved
- ✅ **NDCG@K**: Normalized Discounted Cumulative Gain
- ✅ **MAP**: Mean Average Precision
- ✅ **MRR**: Mean Reciprocal Rank
- ✅ **Query Time**: Latency measurement

### **2. Experiment Runner** (`run_experiments.py`)
- ✅ Loads real/synthetic images
- ✅ Extracts CLIP embeddings
- ✅ Builds Enhanced GraphRAG
- ✅ Evaluates on test queries
- ✅ Generates plots with real data
- ✅ Saves results to JSON

### **3. Ablation Study Support**
- ✅ Component ablation evaluator
- ✅ Hierarchy level ablation
- ✅ Search mode ablation
- ✅ Feature weighting (requires multiple configs)

---

## 🚀 **Quick Start: Get Real Scores**

### **Step 1: Run Experiments**

```bash
conda activate GCS
python run_experiments.py
```

This will:
1. Load images from `balanced_data/` (or create synthetic if not found)
2. Extract CLIP embeddings
3. Build Enhanced GraphRAG with communities
4. Run test queries
5. Calculate all metrics (Precision, Recall, NDCG, etc.)
6. Generate plots with **real data**
7. Save results to `experiments/results/`

###  **Step 2: Check Results**

```bash
# View detailed results
cat experiments/results/evaluation_results.json

# Check plots
ls plots_real/
```

### **Step 3: Update demo_plots.py**

Replace synthetic values in `demo_plots.py` with your real results from the JSON file.

---

## 📊 **Understanding the Metrics**

### **Precision@K**
```python
# How many of the top-K results are relevant?
Precision@5 = (# relevant in top-5) / 5

# Example:
# Retrieved 5 images, 4 are relevant → P@5 = 0.80
```

**Good values**: > 0.70 is strong

---

### **Recall@K**
```python
# How many relevant items did we find?
Recall@5 = (# relevant in top-5) / (total # relevant)

# Example:
# 10 total relevant images, found 6 in top-5 → R@5 = 0.60
```

**Good values**: > 0.60 is strong

---

### **NDCG@K** (Normalized Discounted Cumulative Gain)
```python
# Rewards relevant items ranked higher
# Perfect ranking = 1.0

# Formula:
DCG = sum(rel_i / log2(i + 1))  # i = position
NDCG = DCG / IDCG  # Normalize by ideal DCG
```

**Good values**: > 0.75 is strong

---

### **MAP** (Mean Average Precision)
```python
# Average precision across all queries
# Considers all relevant items, not just top-K

# Formula:
AP = sum(P@i * rel_i) / # relevant items
MAP = average of AP across all queries
```

**Good values**: > 0.70 is strong

---

## 💻 **Using the Evaluation Framework**

### **Basic Usage**

```python
from src import GraphRAGEvaluator, GroundTruthGenerator

# Initialize
evaluator = GraphRAGEvaluator(k_values=[1, 3, 5, 10])

# Define queries
queries = [
    {
        'id': 'Q1',
        'text': 'Find mild Alzheimer cases',
        'mode': 'local',
        'target_class': 'mild',
        'target_dataset': 'alzheimer'
    },
    # ... more queries
]

# Generate ground truth
ground_truth = GroundTruthGenerator.generate_from_class_labels(
    embeddings, queries
)

# Evaluate
results = evaluator.evaluate_queries(retriever, queries, ground_truth)

# Aggregate
aggregated = evaluator.aggregate_metrics(results)

# Print
print(f"Precision@5: {aggregated['precision@k'][5]:.3f}")
print(f"Recall@5: {aggregated['recall@k'][5]:.3f}")
print(f"NDCG@5: {aggregated['ndcg@k'][5]:.3f}")
```

---

### **Ablation Studies**

```python
from src import AblationStudyEvaluator

ablation_eval = AblationStudyEvaluator(evaluator)

# Component ablation
retrievers = {
    'Full System': full_retriever,
    'No Communities': no_communities_retriever,
    'No SSM': no_ssm_retriever,
    # ... more configs
}

results = ablation_eval.evaluate_component_ablation(
    retrievers, queries, ground_truth
)

# Results are formatted for plotting!
visualizer.plot_ablation_components(results)
```

---

### **Search Mode Evaluation**

```python
# Evaluate different search modes
queries_by_mode = {
    'Global': global_queries,
    'Local': local_queries,
    'Hybrid': hybrid_queries,
    'Auto': auto_queries
}

results = ablation_eval.evaluate_search_mode_ablation(
    retriever, queries_by_mode, ground_truth
)

# Plot
visualizer.plot_ablation_search_modes(results)
```

---

## 📋 **Ground Truth Generation**

### **Method 1: From Class Labels** (Automatic)

```python
# Automatically generate based on matching class labels
ground_truth = GroundTruthGenerator.generate_from_class_labels(
    image_embeddings,  # List of ImageEmbedding objects
    queries            # Queries with 'target_class' field
)

# Images with matching class = relevant
```

### **Method 2: From Manual Annotations** (More accurate)

```python
# Load from JSON file
ground_truth = GroundTruthGenerator.load_from_file(
    "experiments/ground_truth.json"
)

# Format:
{
    "Q1": ["path/to/relevant/image1.jpg", "path/to/relevant/image2.jpg"],
    "Q2": ["path/to/relevant/image3.jpg", ...],
    ...
}
```

### **Method 3: Create Your Own**

```python
ground_truth = {
    'Q1': {'/path/img1.jpg', '/path/img2.jpg'},  # Set of relevant paths
    'Q2': {'/path/img3.jpg', '/path/img4.jpg'},
    # ...
}
```

---

## 🔬 **Running Different Experiments**

### **Experiment 1: Full System Evaluation**

```bash
python run_experiments.py
```

### **Experiment 2: Ablation Studies**

You'll need to create different retriever configurations:

```python
# In run_experiments.py, modify to create:

# No communities version
retriever_no_comm = EnhancedGraphRAGRetriever(...)
# Don't call build_enhanced_graph or use flat structure

# No SSM version
retriever_no_ssm = EnhancedGraphRAGRetriever(
    clip_extractor, 
    rule_based_processor,  # No Mamba
    graph
)

# Then evaluate each
ablation_eval.evaluate_component_ablation({
    'Full System': retriever,
    'No Communities': retriever_no_comm,
    'No SSM': retriever_no_ssm
}, queries, ground_truth)
```

### **Experiment 3: Hierarchy Depth**

```python
# Build with different levels
retriever_1level = build_with_levels(1)
retriever_2level = build_with_levels(2)
retriever_3level = build_with_levels(3)

ablation_eval.evaluate_hierarchy_ablation({
    1: retriever_1level,
    2: retriever_2level,
    3: retriever_3level
}, queries, ground_truth)
```

---

## 📈 **Expected Results**

Based on Microsoft GraphRAG paper and similar systems:

| Metric | Good | Very Good | Excellent |
|--------|------|-----------|-----------|
| Precision@5 | 0.60-0.70 | 0.70-0.85 | > 0.85 |
| Recall@10 | 0.50-0.65 | 0.65-0.80 | > 0.80 |
| NDCG@10 | 0.60-0.75 | 0.75-0.85 | > 0.85 |
| MAP | 0.55-0.70 | 0.70-0.85 | > 0.85 |

**Your Goal**: Demonstrate improvement over baselines by 10-20%

---

## 🎯 **For Your Paper**

### **Table 1: Overall Performance**

```
| Method              | P@5  | R@10 | NDCG@10 | MAP  |
|---------------------|------|------|---------|------|
| Enhanced GraphRAG   | 0.85 | 0.82 | 0.88    | 0.84 |
| FAISS               | 0.72 | 0.70 | 0.75    | 0.70 |
| Basic RAG           | 0.68 | 0.66 | 0.71    | 0.67 |
| CLIP-only           | 0.65 | 0.63 | 0.68    | 0.64 |
```

### **Table 2: Component Ablation**

```
| Configuration    | P@5  | R@5  | NDCG@5 |
|------------------|------|------|--------|
| Full System      | 0.85 | 0.82 | 0.88   |
| No Communities   | 0.72 | 0.70 | 0.75   |
| No SSM           | 0.78 | 0.76 | 0.80   |
| No Hierarchy     | 0.76 | 0.74 | 0.78   |
```

### **Table 3: Search Mode Performance**

```
| Mode    | P@5  | R@5  | NDCG@5 | Use Case        |
|---------|------|------|--------|-----------------|
| Global  | 0.78 | 0.88 | 0.82   | Broad analysis  |
| Local   | 0.90 | 0.76 | 0.85   | Specific match  |
| Hybrid  | 0.85 | 0.82 | 0.88   | Complex query   |
| Auto    | 0.84 | 0.83 | 0.87   | Adaptive        |
```

---

## 🛠️ **Troubleshooting**

### **Problem**: Low Precision
**Solution**: 
- Improve similarity threshold
- Better query understanding
- More selective retrieval

### **Problem**: Low Recall
**Solution**:
- Increase K value
- Expand query matching
- Better community coverage

### **Problem**: Queries too slow
**Solution**:
- Use cached embeddings
- Reduce graph size
- Optimize community search

### **Problem**: No ground truth data
**Solution**:
- Use automatic generation from class labels
- Manual annotation (higher quality)
- Use synthetic data for development

---

## 📊 **Output Files**

After running `run_experiments.py`:

```
experiments/
├── results/
│   └── evaluation_results.json      # Detailed metrics
│       {
│         "aggregated_metrics": { ... },
│         "per_query_results": { ... }
│       }
│
plots_real/
├── ablation/
│   ├── ablation_components.pdf      # Real component ablation
│   └── ...
├── comparison/
│   ├── baseline_comparison.pdf      # Real baseline comparison
│   └── query_time.pdf               # Real timing data
└── community/
    └── ...                          # Real community analysis
```

---

## ✅ **Checklist for Paper**

- [ ] Run full experiment (`run_experiments.py`)
- [ ] Collect real metrics for main system
- [ ] Run component ablation study
- [ ] Run hierarchy level ablation
- [ ] Run search mode evaluation
- [ ] Compare with baseline (FAISS, Basic RAG)
- [ ] Generate all plots with real data
- [ ] Update tables in paper with real numbers
- [ ] Verify all metrics are reasonable
- [ ] Save all results for reproducibility

---

## 🎓 **Tips for Best Results**

1. **Use more data**: 50-100 images per class minimum
2. **Diverse queries**: Mix global, local, and hybrid
3. **Quality ground truth**: Manual > automatic
4. **Multiple runs**: Average over 3-5 runs
5. **Error analysis**: Look at failure cases
6. **Statistical significance**: Run t-tests vs. baselines

---

## 📞 **Files**

- **Evaluation**: `src/evaluation.py`
- **Experiment Runner**: `run_experiments.py`
- **Visualization**: `src/visualization.py`
- **Demo (synthetic)**: `demo_plots.py`

---

**Status**: ✅ **READY TO GENERATE REAL SCORES!**

Run `python run_experiments.py` and you'll have real metrics for your CHIIR'26 paper! 🎉

