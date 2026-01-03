# Microsoft GraphRAG-Inspired Approach for Medical Image Retrieval

## 📖 **Overview**

This document explains our adaptation of Microsoft GraphRAG (from [https://github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)) for medical image retrieval, targeting CHIIR'26 submission.

---

## 🎯 **Research Goal**

Develop a novel Graph-based Classification System (GCS) for medical image retrieval that combines:
- **Microsoft GraphRAG methodology** (hierarchical communities + global/local search)
- **Multimodal embeddings** (CLIP for images + text)
- **State-space models** (Mamba for query processing)
- **Medical domain adaptation** (disease-based + visual feature clustering)

---

## 📚 **Microsoft GraphRAG Background**

### Original GraphRAG (Text-based):

Microsoft's GraphRAG ([paper: "From Local to Global"](https://github.com/microsoft/graphrag)) introduces a **modular graph-based RAG system** with:

1. **Entity Extraction**: LLM identifies entities and relationships from text documents
2. **Knowledge Graph Construction**: Build graph from extracted entities
3. **Community Detection**: Hierarchical clustering using Leiden algorithm
4. **Community Summarization**: Generate summaries at each hierarchy level
5. **Global Search**: Map-reduce over community summaries (broad questions)
6. **Local Search**: Entity-level traversal (specific questions)

### Key Innovation:
**From Local to Global**: Move from document-level understanding to community-level patterns, enabling better handling of broad analytical queries.

---

## 🔄 **Our Adaptation for Medical Images**

### Challenge:
Microsoft GraphRAG is designed for **text documents**, but we have **medical images** with limited metadata.

### Solution:
We adapted the methodology for **multimodal medical data** by:

| Microsoft GraphRAG (Text) | Our Adaptation (Medical Images) |
|--------------------------|----------------------------------|
| **Entity Extraction** (NER from text) | **Visual Feature Extraction** (CLIP embeddings + medical concepts) |
| **Text Relationships** (co-occurrence) | **Visual Similarity** + Disease relationships |
| **Topical Communities** (e.g., "climate change") | **Medical Communities** (disease → visual → severity) |
| **Text Embeddings** | **Multimodal Embeddings** (CLIP image-text) |
| **LLM Summarization** | **SSM (Mamba) Summarization** |

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────┐
│              USER QUERY                                  │
│  e.g., "Find similar Alzheimer's cases"                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│           SSM QUERY PROCESSOR (Mamba 1.4B)               │
│  - Intent detection: analysis/retrieval/comparison       │
│  - Query understanding and expansion                     │
│  - Medical concept extraction                            │
└──────────────────┬───────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
    GLOBAL SEARCH      LOCAL SEARCH       ← Search Mode Selection
         │                   │
         │                   │
    ┌────▼────┐         ┌────▼────┐
    │Community│         │ Entity  │
    │Summaries│         │Matching │
    └────┬────┘         └────┬────┘
         │                   │
         └─────────┬─────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│     HIERARCHICAL MEDICAL KNOWLEDGE GRAPH                 │
│                                                           │
│  ┌─────────────────────────────────────────────┐        │
│  │ Level 0 (Global): Disease Categories         │        │
│  │  - Alzheimer's community (10 cases)          │        │
│  │  - Brain Tumor community (10 cases)          │        │
│  │  - Parkinson's community (10 cases)          │        │
│  └──────────────┬────────────────────────────────┘       │
│                 │                                         │
│  ┌──────────────▼────────────────────────────────┐       │
│  │ Level 1 (Mid): Visual Feature Groups          │       │
│  │  - Alzheimer's: mild pattern group (7 cases)  │       │
│  │  - Brain Tumor: glioma pattern (7 cases)      │       │
│  │  - Parkinson's: early stage (6 cases)         │       │
│  └──────────────┬────────────────────────────────┘       │
│                 │                                         │
│  ┌──────────────▼────────────────────────────────┐       │
│  │ Level 2 (Local): Fine-grained Classes          │       │
│  │  - Mild Alzheimer's (3 cases)                  │       │
│  │  - Glioma Grade II (4 cases)                   │       │
│  │  - Parkinson's Stage 1 (3 cases)               │       │
│  └─────────────────────────────────────────────────┘      │
│                                                           │
└──────────────────┬────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│         MULTIMODAL EMBEDDING LAYER                       │
│  - CLIP embeddings (image ↔ text)                       │
│  - Medical feature vectors                               │
│  - Community centroid embeddings                         │
└──────────────────┬────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│         BALANCED MEDICAL DATASETS                        │
│  - Alzheimer's: Mild/Moderate/Severe/Non-demented       │
│  - Brain Tumor: Glioma/Meningioma/Pituitary/No-tumor    │
│  - Parkinson's: Normal/Parkinson                         │
│  - MS: Normal/MS                                         │
└──────────────────────────────────────────────────────────┘
```

---

## 🔬 **Hierarchical Community Detection**

### Level 0: Disease Type Communities (Global)
**Purpose**: Group images by primary disease category

**Method**:
- Cluster by dataset (Alzheimer, Brain Tumor, Parkinson, MS)
- Create broad disease communities
- Weight: **100% disease-based**

**Example Communities**:
- `L0_C0_alzheimer`: All Alzheimer's cases (10 images)
- `L0_C1_brain_tumor`: All Brain Tumor cases (10 images)

**Use Case**: Global queries like "What patterns exist across all neurological diseases?"

---

### Level 1: Visual Feature Communities (Mid)
**Purpose**: Group images by visual similarity within disease types

**Method**:
- Within each Level 0 community, perform **Agglomerative Clustering**
- Based on CLIP embedding similarity
- Weight: **Disease (60%) + Visual Features (40%)**

**Example Communities**:
- `L1_C0`: Alzheimer's with mild atrophy patterns (7 cases)
- `L1_C2`: Brain Tumor with glioma characteristics (7 cases)

**Use Case**: Mid-level queries like "Find visually similar tumor patterns"

---

### Level 2: Fine-grained Communities (Local)
**Purpose**: Highly specific case grouping by severity/stage

**Method**:
- Within Level 1 communities, cluster by **class labels** (mild/moderate/severe)
- Homogeneous groups for precise matching
- Weight: **Class labels (100%)**

**Example Communities**:
- `L2_C0`: Mild Alzheimer's only (3 cases)
- `L2_C1`: Glioma Grade II only (4 cases)

**Use Case**: Local queries like "Find cases identical to this mild Alzheimer's scan"

---

## 📊 **Community Summarization**

Each community gets an **automated summary** generated by SSM (Mamba):

### Level 0 Summary Example:
```
"Global medical community for Alzheimer dataset containing 10 cases.
Distribution: 4 mild, 3 moderate, 3 severe. This community represents
the overall disease category with diverse presentations."
```

### Level 1 Summary Example:
```
"Mid-level Alzheimer community with 7 cases showing predominantly mild
patterns. Predominantly mild (42.9%), indicating a visually coherent
subgroup within the broader Alzheimer category."
```

### Level 2 Summary Example:
```
"Fine-grained community of 3 Alzheimer cases specifically classified as
'mild'. This represents a homogeneous group with consistent clinical
presentations, ideal for detailed case-by-case analysis."
```

---

## 🔍 **Search Strategies**

### 1. **Global Search** (Microsoft GraphRAG Map-Reduce)

**When**: Broad, analytical queries
- "What patterns exist across all diseases?"
- "Compare neurological conditions"
- "Summarize all tumor types"

**How**:
1. Find relevant **Level 0 & 1 communities** based on query
2. **Map phase**: Retrieve community summaries in parallel
3. **Reduce phase**: Aggregate insights across communities
4. Select **representative cases** from matched communities

**Output**:
- Community-level insights
- Representative images from multiple communities
- High-level patterns and trends

---

### 2. **Local Search** (Entity-Level Precision)

**When**: Specific, retrieval-focused queries
- "Find scans similar to this mild Alzheimer's case"
- "Retrieve glioma cases"
- "Show me specific tumor examples"

**How**:
1. Generate query embedding (CLIP)
2. Direct **cosine similarity** search across all image nodes
3. Return **top-K most similar** cases
4. Provide fine-grained matches

**Output**:
- Highly similar individual cases
- Precise matches with similarity scores
- Entity-level details

---

### 3. **Hybrid Search** (Combined)

**When**: Complex queries requiring both breadth and depth
- "Compare Alzheimer's progression patterns"
- "Find tumor cases and analyze trends"

**How**:
1. Run **Global search** for community-level context
2. Run **Local search** for specific case matches
3. **Merge** results (deduplicate)
4. Combine reasoning from both strategies

**Output**:
- Global context + Local precision
- Best of both worlds

---

### 4. **Auto Mode** (Intelligent Selection)

**When**: User doesn't specify search mode

**How**:
SSM (Mamba) detects query intent:
- **Analysis/Comparison** → Global search
- **Retrieval/Similarity** → Local search
- **Classification/Complex** → Hybrid search

---

## 🧠 **Query Processing with SSM (Mamba)**

### Why Mamba?
- **State-space models** handle long-range dependencies
- Efficient for medical query understanding
- Can process both queries and generate summaries

### SSM Role:
1. **Intent Detection**: Classify query type (analysis vs. retrieval)
2. **Query Understanding**: Extract medical concepts
3. **Community Summarization**: Generate textual descriptions
4. **Context Assembly**: Build retrieval context

### Fallback:
- Rule-based processing if Mamba unavailable
- Ensures system always functions

---

## 🎯 **Novel Contributions**

### 1. **First Multimodal Adaptation of Microsoft GraphRAG**
- Extended text-based GraphRAG to **medical images**
- Novel combination: disease + visual features

### 2. **Hierarchical Medical Communities**
- **3-level structure**: Disease → Visual → Fine-grained
- Tailored for clinical use cases

### 3. **State-Space Model Integration**
- Mamba (1.4B) for query processing
- SSM for community summarization
- Demonstrates SSM potential beyond sequence modeling

### 4. **Flexible Retrieval Paradigm**
- Global/Local/Hybrid/Auto modes
- Adapts to query complexity

### 5. **Balanced Medical Data**
- Bias-reduced sampling
- Equal representation across classes

---

## 📈 **Comparison with Baseline RAG**

| Feature | Baseline RAG | Our Enhanced GraphRAG |
|---------|--------------|----------------------|
| **Structure** | Flat vector search | 3-level hierarchical communities |
| **Search** | Single-mode similarity | Global/Local/Hybrid/Auto |
| **Context** | No community awareness | Community summaries at each level |
| **Reasoning** | Direct match | Multi-hop through communities |
| **Scalability** | All vectors scanned | Community-based indexing |
| **Explainability** | Similarity scores | Reasoning paths + community context |

---

## 📊 **Evaluation Plan for CHIIR'26**

### Metrics:
1. **Retrieval Quality**:
   - Precision@K, Recall@K, NDCG@K
   - Compare Global vs. Local vs. Baseline

2. **Community Coherence**:
   - Silhouette score per level
   - Intra-community vs. inter-community distance

3. **Search Mode Appropriateness**:
   - User study: Does auto-selection match user intent?

4. **Explainability**:
   - Human evaluation of reasoning paths
   - Community summary usefulness

5. **Efficiency**:
   - Query latency (global vs. local)
   - Memory footprint

### Baselines:
1. **Traditional Vector Search** (FAISS)
2. **Basic RAG** (no communities)
3. **CLIP-only retrieval**
4. **Previous GraphRAG** (pre-enhancement)

---

## 🔧 **Technical Implementation**

### Files:
- `src/community_detection.py`: Hierarchical clustering (Agglomerative + Disease-based)
- `src/community_summarization.py`: SSM-based summarization
- `src/enhanced_graphrag.py`: Global/Local/Hybrid retrieval
- `src/ssm.py`: Mamba query processor
- `src/clip_embeddings.py`: CLIP multimodal embeddings

### Dependencies:
```bash
# Core
torch, transformers (Mamba + CLIP)
networkx (graph structure)
scikit-learn (clustering)

# Community Detection
python-louvain (Louvain/Leiden-style)
scipy, faiss-cpu
```

### Usage:
```python
from src import EnhancedGraphRAGRetriever

# Initialize
retriever = EnhancedGraphRAGRetriever(clip, ssm, graph)

# Build graph with communities
retriever.build_enhanced_graph(embeddings)

# Retrieve with auto mode
result = retriever.retrieve("Find Alzheimer's cases", top_k=10, search_mode="auto")
```

---

## 🎓 **For CHIIR'26 Paper**

### Title Suggestions:
1. "Hierarchical GraphRAG for Medical Image Retrieval: A Microsoft GraphRAG Adaptation"
2. "From Local to Global in Medical Imaging: Community-Based Retrieval with State-Space Models"
3. "Enhanced GraphRAG: Hierarchical Community Detection for Medical Image Retrieval"

### Key Claims:
- ✅ **Novel adaptation** of Microsoft GraphRAG for medical images
- ✅ **First integration** of SSM (Mamba) with GraphRAG
- ✅ **Hierarchical organization** improves retrieval quality
- ✅ **Flexible search modes** adapt to query complexity
- ✅ **Explainable reasoning** through community context

### Contributions:
1. Methodology for adapting text-based GraphRAG to images
2. Hierarchical community detection for medical domains
3. Integration of state-space models for query processing
4. Evaluation framework for community-based retrieval

---

## ✅ **Current Status**

- ✅ Core implementation complete
- ✅ Hierarchical communities working
- ✅ Global/Local/Hybrid search functional
- ✅ SSM integration operational
- ✅ Demo successfully tested
- ⏳ Baseline comparison pending
- ⏳ Full evaluation pending

---

## 📚 **References**

1. **Microsoft GraphRAG**: https://github.com/microsoft/graphrag
2. **"From Local to Global" Paper**: Microsoft Research (Arxiv)
3. **Mamba (SSM)**: https://huggingface.co/state-spaces/mamba-1.4b-hf
4. **CLIP**: https://huggingface.co/docs/transformers/model_doc/clip

---

**Last Updated**: October 10, 2025  
**Target**: CHIIR'26 Submission  
**Status**: Implementation complete, ready for evaluation

