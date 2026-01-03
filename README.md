# Graph-based Classification System (GCS) for Medical Image Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Microsoft GraphRAG-Inspired Medical Image Retrieval System**

---

## 🎯 **Project Overview**

This project implements a **novel Graph-based Classification System (GCS)** for medical image retrieval, adapting [Microsoft GraphRAG](https://github.com/microsoft/graphrag) methodology for multimodal medical data. The system combines hierarchical community detection, state-space models (Mamba), and CLIP embeddings for intelligent medical image search.

### **Key Innovation**: 
First adaptation of Microsoft's "From Local to Global" GraphRAG approach for medical imaging, enabling both broad pattern analysis and precise case matching.

---

## 🏗️ **Architecture**

```
User Query → SSM (Mamba) Query Processing → Search Mode Selection
                                              ├─ Global Search (Community-level)
                                              ├─ Local Search (Entity-level)
                                              └─ Hybrid Search (Combined)
                                                    ↓
                      Hierarchical Medical Knowledge Graph
                          Level 0: Disease Categories
                          Level 1: Visual Feature Groups
                          Level 2: Fine-grained Cases
                                                    ↓
                        CLIP Multimodal Embeddings
                                                    ↓
                        Image Data Dump (any structure)
```

---

## ✨ **Features**

### 🔬 **Microsoft GraphRAG Adaptations**
- ✅ **Hierarchical Community Detection**: 3-level structure (Disease → Visual → Fine-grained)
- ✅ **Global Search**: Map-reduce over community summaries for broad queries
- ✅ **Local Search**: Entity-level precision for specific queries
- ✅ **Hybrid Search**: Combined approach for complex queries
- ✅ **Auto Mode**: Intelligent search strategy selection

### 🧠 **State-Space Model Integration**
- ✅ **Mamba 1.4B**: Query processing and intent detection
- ✅ **Community Summarization**: Automated medical descriptions
- ✅ **Rule-based Fallback**: System works without neural models

### 🎨 **Multimodal Embeddings**
- ✅ **CLIP**: Image-text cross-modal understanding
- ✅ **Hugging Face Integration**: Transformers-based implementation
- ✅ **Fallback System**: Rule-based embeddings if needed

### 📊 **Medical Domain Specifics**
- ✅ **4 Disease Datasets**: Alzheimer's, Brain Tumor, Parkinson's, MS
- ✅ **Balanced Sampling**: Bias-reduced data distribution
- ✅ **Hierarchical Organization**: Clinically meaningful grouping

### 🧪 **Team Testing**
- ✅ **Data dump support**: index any directory layout (labeled or unlabeled)
- ✅ **Low-code CLI**: build artifacts, run queries, propagate tags

---

## 📁 **Project Structure**

```
GCS/
├── src/                          # Source code
│   ├── clip_embeddings.py        # CLIP embedding extraction
│   ├── data_dump.py              # Index arbitrary data-dump folders
│   ├── ssm.py                    # Mamba SSM query processor
│   ├── graphRAG.py               # Basic GraphRAG implementation
│   ├── community_detection.py    # Hierarchical clustering
│   ├── community_summarization.py # SSM-based summarization
│   ├── enhanced_graphrag.py      # Enhanced retriever (Global/Local/Hybrid)
│   ├── data_utils.py             # Data loading utilities
│   └── model_config.py           # Model configurations
├── tests/                        # Test suite
│   ├── test_enhanced_graphrag.py # Enhanced GraphRAG tests
│   └── test_mamba_integration.py # Mamba integration tests
├── scripts/                      # Utility scripts
│   ├── gcs_cli.py                # Team testing CLI (build/query/tag)
│   └── balanced_sampling.py      # Dataset balancing
├── docs/                         # Documentation
│   ├── README.md                 # Additional docs index
│   └── GRAPHRAG_APPROACH.md      # Detailed methodology
├── data/                         # Medical datasets
│   ├── AlzheimerDataset/
│   ├── brain-tumor-mri-dataset/
│   ├── parkinsons_dataset_processed/
│   └── ms_slices_central/
├── balanced_data/                # Balanced sampled datasets
├── requirements.txt              # Dependencies
└── run_tests.py                  # Test runner
```

---

## 🚀 **Quick Start**

### 1. **Installation**

```bash
# Clone repository
git clone <repository-url>
cd GCS

# Create conda environment
conda create -n GCS python=3.8
conda activate GCS

# Install dependencies
pip install -r requirements.txt
```

### 2. **Team Testing (Any Data Dump)**

Build reusable artifacts from any folder structure (labeled or unlabeled):

```bash
python scripts/gcs_cli.py build --data-root data --data-root balanced_data --artifact artifacts/gcs_artifacts.pkl
```

Then run text and/or image queries:

```bash
python scripts/gcs_cli.py query --artifact artifacts/gcs_artifacts.pkl --text "similar MRI scans" --mode auto --top-k 10
```

Single-shot tagging (propagate to nearest neighbors in embedding space):

```bash
python scripts/gcs_cli.py tag --artifact artifacts/gcs_artifacts.pkl --image /path/to/seed.jpg --tag "review_me" --k 50
```

More options: `docs/TEAM_TESTING.md`.

### 3. **Run Full Test**

```bash
python run_tests.py
```

### 4. **Use in Your Code**

```python
from src import (
    EnhancedGraphRAGRetriever,
    CLIPEmbeddingExtractor,
    SSMQueryProcessor,
    MedicalKnowledgeGraph
)

# Initialize components
clip_extractor = CLIPEmbeddingExtractor()
ssm_processor = SSMQueryProcessor(model_key="mamba-1.4b")
graph = MedicalKnowledgeGraph()

# Create enhanced retriever
retriever = EnhancedGraphRAGRetriever(
    clip_extractor=clip_extractor,
    ssm_processor=ssm_processor,
    graph=graph
)

# Build graph with your embeddings
retriever.build_enhanced_graph(image_embeddings)

# Retrieve with auto mode
result = retriever.retrieve(
    query="Find similar Alzheimer's cases",
    top_k=10,
    search_mode="auto"  # or "global", "local", "hybrid"
)

# Access results
print(f"Mode: {result.search_mode}")
print(f"Retrieved: {len(result.retrieved_images)} images")
print(f"Communities: {len(result.community_context)}")
print(f"Reasoning: {result.reasoning_path}")
```

---

## 📊 **Datasets**

The repository includes example medical datasets under `data/` and `balanced_data/`, but the team-testing CLI also supports arbitrary folders (see `docs/TEAM_TESTING.md`).

### **Example Medical Datasets**:

1. **Alzheimer's Dataset** (86,437 images)
   - Classes: Non Demented, Very Mild, Mild, Moderate
   - Source: MRI brain scans

2. **Brain Tumor MRI Dataset** (7,023 images)
   - Classes: Glioma, Meningioma, Pituitary, No Tumor
   - Source: MRI brain scans

3. **Parkinson's Dataset** (900 images)
   - Classes: Normal, Parkinson
   - Source: Spiral drawing images

4. **MS Dataset** (420 images)
   - Classes: Normal, MS
   - Source: MRI brain scans

### **Balanced Sampling**:
Each dataset is balanced to ensure equal representation of classes, reducing bias in retrieval.

---

## 🔍 **Search Modes**

### **1. Global Search** (Broad Analysis)
**Use Case**: "What patterns exist across all neurological diseases?"

**How it Works**:
1. Find relevant communities at Level 0/1
2. Retrieve community summaries
3. Map-reduce across communities
4. Return representative cases

**Best For**: Exploratory analysis, pattern discovery

---

### **2. Local Search** (Precise Matching)
**Use Case**: "Find scans similar to this mild Alzheimer's case"

**How it Works**:
1. Generate query embedding (CLIP)
2. Direct cosine similarity search
3. Return top-K most similar images
4. Entity-level precision

**Best For**: Specific case retrieval, similarity search

---

### **3. Hybrid Search** (Combined)
**Use Case**: "Compare Alzheimer's progression patterns"

**How it Works**:
1. Global search for broad context
2. Local search for specific matches
3. Merge and deduplicate results
4. Combine reasoning paths

**Best For**: Complex queries requiring breadth + depth

---

### **4. Auto Mode** (Intelligent Selection)
**Use Case**: Any query (system decides)

**How it Works**:
1. SSM analyzes query intent
2. Selects appropriate search mode:
   - Analysis/Comparison → Global
   - Retrieval/Similarity → Local
   - Classification/Complex → Hybrid

**Best For**: General use, uncertain query types

---

## 🧪 **Evaluation**

### **Metrics** (Planned):
- **Retrieval Quality**: Precision@K, Recall@K, NDCG@K
- **Community Coherence**: Silhouette scores per level
- **Search Mode Appropriateness**: User study validation
- **Explainability**: Human evaluation of reasoning
- **Efficiency**: Query latency, memory footprint

### **Baselines** (Planned):
1. Traditional Vector Search (FAISS)
2. Basic RAG (no communities)
3. CLIP-only retrieval
4. Previous GraphRAG (pre-enhancement)

---

## 📚 **Documentation**

- **`docs/GRAPHRAG_APPROACH.md`**: Comprehensive methodology explanation
- **`docs/TEAM_TESTING.md`**: Low-code build/query/tag workflow
- **Code Documentation**: Inline docstrings in all modules

---

## 🛠️ **Key Dependencies**

```txt
# Core ML
torch>=2.0.0
transformers>=4.39.0  # For Mamba + CLIP

# Graph & RAG
networkx>=3.0
langchain>=0.3.0

# Community Detection
python-louvain>=0.16
scipy>=1.10.0
faiss-cpu>=1.7.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=9.5.0
```

---

## 🎓 **Research Contributions**

### **Novel Aspects**:
1. **First multimodal adaptation** of Microsoft GraphRAG
2. **Hierarchical medical communities** (disease → visual → fine-grained)
3. **State-space model integration** with GraphRAG
4. **Flexible retrieval paradigm** (Global/Local/Hybrid/Auto)
5. **Bias-reduced medical image retrieval**

---

## 🤝 **Citation**

Cite the original Microsoft GraphRAG:

```bibtex
@misc{graphrag2024,
  title={GraphRAG: A Modular Graph-based Retrieval-Augmented Generation System},
  author={Microsoft Research},
  url={https://github.com/microsoft/graphrag},
  year={2024}
}
```

---

## 📧 **Contact & Support**

- **Repository**: [Link to repo]
- **Issues**: Use GitHub Issues for bug reports
- **Questions**: Open GitHub Discussions

---

## 🔒 **License**

MIT License - See LICENSE file for details

---

## 🙏 **Acknowledgments**

- **Microsoft GraphRAG Team**: For the original GraphRAG methodology
- **Hugging Face**: For transformers library (Mamba + CLIP)
- **Medical Dataset Providers**: For open-source medical imaging data

---

## 📈 **Project Status**

| Component | Status |
|-----------|--------|
| CLIP Embeddings | ✅ Complete |
| SSM (Mamba) Integration | ✅ Complete |
| Basic GraphRAG | ✅ Complete |
| Community Detection | ✅ Complete |
| Community Summarization | ✅ Complete |
| Enhanced Retrieval (Global/Local/Hybrid) | ✅ Complete |
| Demo | ✅ Working |
| Tests | ✅ Passing |
| Baseline Implementation | ⏳ Pending |
| Evaluation Framework | ⏳ Pending |
| Paper | ⏳ In Progress |

---

## 🎯 **Roadmap**

- [x] Implement Microsoft GraphRAG-inspired features
- [x] Hierarchical community detection (3 levels)
- [x] Global/Local/Hybrid search modes
- [x] SSM integration for query processing
- [x] CLIP multimodal embeddings
- [ ] Baseline implementations (FAISS, Basic RAG)
- [ ] Comprehensive evaluation framework
- [ ] User study for explainability
- [ ] Paper submission

---

**Last Updated**: October 10, 2025  
**Version**: 1.0.0  
**Status**: Core implementation complete, ready for evaluation

---

<p align="center">
  <strong>🎉 Microsoft GraphRAG meets Medical Imaging 🏥</strong>
</p>
