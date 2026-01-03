# 🏗️ System Architecture Overview - How Everything Works Together

**Your GraphRAG Medical Image Retrieval System**

---

## 📋 **HIGH-LEVEL ARCHITECTURE**

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│         "Find mild Alzheimer cases with atrophy"                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING (SSM/Mamba)                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Intent Detection: "Find" → Local Search              │   │
│  │  2. Entity Extraction: "mild Alzheimer", "atrophy"       │   │
│  │  3. Search Mode Selection: Global/Local/Hybrid           │   │
│  │  4. Generate Text Embedding: CLIP text encoder           │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              HIERARCHICAL KNOWLEDGE GRAPH                        │
│                                                                   │
│  Level 0 (Global): Disease Communities                           │
│  ┌─────────────┬──────────────┬──────────────┬────────────┐    │
│  │ Alzheimer   │ Brain Tumor  │ Parkinson's  │    MS      │    │
│  │ (100 imgs)  │ (100 imgs)   │ (100 imgs)   │ (100 imgs) │    │
│  └──────┬──────┴──────┬───────┴──────┬───────┴─────┬──────┘    │
│         │             │              │             │             │
│  Level 1 (Mid): Visual Similarity Clusters                       │
│  ┌──────▼──────┬──────▼──────┬───────▼────┬────────▼──────┐    │
│  │ L1_C0       │ L1_C1       │ L1_C2      │ L1_C3         │    │
│  │ (Mild)      │ (Moderate)  │ (Severe)   │ (Early)       │    │
│  │ 15 imgs     │ 12 imgs     │ 10 imgs    │ 13 imgs       │    │
│  └──────┬──────┴──────┬──────┴────────┬───┴───────┬───────┘    │
│         │             │               │           │              │
│  Level 2 (Local): Fine-grained Classes                           │
│  ┌──────▼──────┬──────▼──────┬────────▼────┬──────▼──────┐    │
│  │ L2_C0       │ L2_C1       │ L2_C2       │ L2_C3       │    │
│  │ (Specific)  │ (Specific)  │ (Specific)  │ (Specific)  │    │
│  │ 5-8 imgs    │ 4-7 imgs    │ 6-9 imgs    │ 5-8 imgs    │    │
│  └─────────────┴─────────────┴─────────────┴─────────────┘    │
│                                                                   │
│  Total: 44 Communities across 3 Levels                           │
│  Nodes: 400 images + metadata                                    │
│  Edges: 75,625 similarity connections                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-STRATEGY RETRIEVAL                        │
│                                                                   │
│  If Mode = GLOBAL:                                               │
│    → Search community summaries                                  │
│    → Return top-K communities                                    │
│    → Aggregate images from communities                           │
│                                                                   │
│  If Mode = LOCAL:                                                │
│    → Direct similarity search in embeddings                      │
│    → Return top-K most similar images                            │
│                                                                   │
│  If Mode = HYBRID:                                               │
│    → Combine community + similarity                              │
│    → Weighted ranking                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RANKING & RESULTS                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Top-K Images (K=1,3,5,10):                              │   │
│  │    1. patient_123.jpg (score: 0.95)                      │   │
│  │       Community: L0_C0_alzheimer → L1_C2 → L2_C5         │   │
│  │       Explanation: "Mild atrophy pattern"                │   │
│  │                                                           │   │
│  │    2. patient_456.jpg (score: 0.89)                      │   │
│  │       Community: L0_C0_alzheimer → L1_C2 → L2_C5         │   │
│  │       Explanation: "Similar hippocampal changes"         │   │
│  │                                                           │   │
│  │    3. patient_789.jpg (score: 0.85)                      │   │
│  │       ...                                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ **DATA STORAGE STRUCTURE**

### **1. File System Storage**

```
GCS/
│
├── balanced_data/                    # Raw Images
│   ├── balanced_alzheimer/
│   │   ├── mild dementia/
│   │   │   ├── alzheimer_001.jpg    ← Original images
│   │   │   ├── alzheimer_002.jpg
│   │   │   └── ...
│   │   ├── moderate/
│   │   └── severe/
│   ├── balanced_brain_tumor/
│   ├── balanced_parkinson/
│   └── balanced_ms/
│
├── embeddings_cache/                 # Pre-computed Embeddings
│   ├── alzheimer_embeddings.pkl      ← CLIP features (512-dim)
│   ├── brain_tumor_embeddings.pkl    ← Cached for speed
│   ├── parkinson_embeddings.pkl
│   └── ms_embeddings.pkl
│
└── experiments/
    └── results/
        └── evaluation_results_enhanced.json  ← Performance metrics
```

### **2. In-Memory Data Structures**

```python
# When system runs, it loads everything into memory:

system_state = {
    # 1. Image Embeddings (400 images × 512 dimensions)
    'image_embeddings': [
        ImageEmbedding(
            image_path='balanced_data/alzheimer/.../img_001.jpg',
            embedding=np.array([0.12, -0.45, ...]),  # 512-dim vector
            class_label='mild dementia',
            dataset='alzheimer'
        ),
        ...  # 400 total
    ],
    
    # 2. Knowledge Graph (NetworkX graph object)
    'graph': {
        'nodes': {
            'img_001': {
                'embedding': np.array([...]),
                'class': 'mild dementia',
                'dataset': 'alzheimer',
                'community_l0': 'L0_C0_alzheimer',
                'community_l1': 'L1_C2',
                'community_l2': 'L2_C5'
            },
            ...  # 400 nodes
        },
        'edges': [
            ('img_001', 'img_002', {'weight': 0.87}),  # Similarity
            ('img_001', 'img_045', {'weight': 0.76}),
            ...  # 75,625 edges
        ]
    },
    
    # 3. Hierarchical Communities (44 total)
    'communities': {
        'level_0': {
            'L0_C0_alzheimer': {
                'members': ['img_001', 'img_002', ..., 'img_100'],
                'centroid': np.array([...]),
                'summary': 'Alzheimer disease patterns with varying severity'
            },
            'L0_C1_brain_tumor': {...},
            'L0_C2_parkinson': {...},
            'L0_C3_ms': {...}
        },
        'level_1': {
            'L1_C0': {
                'parent': 'L0_C0_alzheimer',
                'members': ['img_001', 'img_015', ..., 'img_023'],
                'centroid': np.array([...]),
                'summary': 'Mild cognitive decline subgroup'
            },
            'L1_C1': {...},
            ...  # ~16 communities
        },
        'level_2': {
            'L2_C0': {
                'parent': 'L1_C0',
                'members': ['img_001', 'img_004', 'img_007'],
                'centroid': np.array([...]),
                'summary': 'Early hippocampal atrophy'
            },
            ...  # ~24 communities
        }
    },
    
    # 4. Node Embeddings Index (for fast lookup)
    'node_embeddings': {
        'img_001': np.array([0.12, -0.45, ...]),  # 512-dim
        'img_002': np.array([0.34, 0.23, ...]),
        ...  # 400 entries
    },
    
    # 5. Metadata Index
    'node_metadata': {
        'img_001': {
            'path': 'balanced_data/alzheimer/.../img_001.jpg',
            'class': 'mild dementia',
            'dataset': 'alzheimer',
            'communities': ['L0_C0_alzheimer', 'L1_C2', 'L2_C5']
        },
        ...  # 400 entries
    }
}
```

---

## 🔄 **QUERY PROCESSING FLOW**

### **Step-by-Step: What Happens When User Queries**

```
USER: "Find mild Alzheimer cases with atrophy"
  │
  │ 1. Query enters system
  ▼
┌─────────────────────────────────────────────────────────────┐
│ SSM/MAMBA QUERY PROCESSOR (src/ssm.py)                      │
│                                                              │
│ A. Intent Detection (Rule-based currently):                 │
│    - Scans for keywords: "find", "show", "compare", "all"   │
│    - Result: "find" → Local Search Mode                     │
│                                                              │
│ B. Entity Extraction:                                       │
│    - Extracts: ["mild", "Alzheimer", "atrophy"]            │
│    - Maps to: disease="alzheimer", severity="mild"          │
│                                                              │
│ C. Query Embedding Generation:                              │
│    query_text = "Find mild Alzheimer cases with atrophy"    │
│    query_embedding = CLIP.get_text_features(query_text)     │
│    → Result: 512-dim vector [0.23, -0.12, 0.45, ...]       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ RETRIEVAL STRATEGY SELECTION (src/enhanced_graphrag.py)     │
│                                                              │
│ Based on intent:                                            │
│   IF "find/specific" → LOCAL SEARCH                         │
│   IF "all/show/compare" → GLOBAL SEARCH                     │
│   ELSE → HYBRID SEARCH                                      │
│                                                              │
│ Selected: LOCAL SEARCH                                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ LOCAL SEARCH EXECUTION                                       │
│                                                              │
│ 1. Compute Similarity:                                      │
│    for each image_emb in node_embeddings:                   │
│        similarity = cosine(query_embedding, image_emb)      │
│                                                              │
│ 2. Sort by Similarity:                                      │
│    [                                                         │
│      (img_045, 0.92),  ← Most similar                       │
│      (img_012, 0.89),                                        │
│      (img_078, 0.87),                                        │
│      (img_023, 0.85),                                        │
│      (img_091, 0.83),                                        │
│      ...                                                     │
│    ]                                                         │
│                                                              │
│ 3. Get Top-K (K=5):                                         │
│    top_results = results[:5]                                │
│                                                              │
│ 4. Enrich with Context:                                     │
│    For each result:                                         │
│      - Load image metadata                                  │
│      - Get community path (L0 → L1 → L2)                   │
│      - Generate explanation                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ RESULT RANKING & EXPLANATION                                 │
│                                                              │
│ Results = [                                                  │
│   {                                                          │
│     'rank': 1,                                              │
│     'image': 'img_045',                                     │
│     'path': 'balanced_data/.../alzheimer_045.jpg',          │
│     'similarity': 0.92,                                     │
│     'class': 'mild dementia',                               │
│     'dataset': 'alzheimer',                                 │
│     'community_path': [                                     │
│       'L0_C0_alzheimer',                                    │
│       'L1_C2',                                              │
│       'L2_C5'                                               │
│     ],                                                      │
│     'explanation': 'From mild cognitive decline subgroup    │
│                     showing hippocampal atrophy patterns',  │
│     'confidence': 0.92                                      │
│   },                                                         │
│   { rank: 2, ... },                                         │
│   { rank: 3, ... },                                         │
│   { rank: 4, ... },                                         │
│   { rank: 5, ... }                                          │
│ ]                                                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ RETURN TO USER                                               │
│                                                              │
│ Display: Top-5 Images with Explanations                     │
│ Time: 71ms                                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 **MAMBA/SSM COMMUNICATION FLOW**

### **Current Implementation**

```
┌─────────────────────────────────────────────────────────────┐
│ SSMQueryProcessor Class (src/ssm.py)                        │
│                                                              │
│ Currently implements TWO modes:                             │
│                                                              │
│ 1. RULE-BASED MODE (Active by default):                    │
│    ┌──────────────────────────────────────────────────┐    │
│    │ def process_query(query_text):                   │    │
│    │     # Keyword matching                            │    │
│    │     if "find" in query or "specific" in query:   │    │
│    │         intent = "local_search"                   │    │
│    │     elif "all" in query or "show" in query:      │    │
│    │         intent = "global_search"                  │    │
│    │     else:                                         │    │
│    │         intent = "hybrid_search"                  │    │
│    │                                                   │    │
│    │     return QueryResult(                           │    │
│    │         intent=intent,                            │    │
│    │         entities=extract_entities(query_text),    │    │
│    │         embedding=clip.get_text_features(...)     │    │
│    │     )                                             │    │
│    └──────────────────────────────────────────────────┘    │
│                                                              │
│ 2. MAMBA MODE (Available but not active):                  │
│    ┌──────────────────────────────────────────────────┐    │
│    │ def _load_model():                                │    │
│    │     from transformers import MambaForCausalLM     │    │
│    │     model = MambaForCausalLM.from_pretrained(     │    │
│    │         "state-spaces/mamba-130m-hf"              │    │
│    │     )                                             │    │
│    │                                                   │    │
│    │ def generate_with_mamba(query):                   │    │
│    │     prompt = f"Analyze query: {query}\n"         │    │
│    │     prompt += "Intent (local/global/hybrid): "    │    │
│    │     response = model.generate(prompt)             │    │
│    │     # Parse Mamba output                          │    │
│    │     return parsed_intent                          │    │
│    └──────────────────────────────────────────────────┘    │
│                                                              │
│ USED BY: EnhancedGraphRAGRetriever                          │
└─────────────────────────────────────────────────────────────┘
```

### **How Mamba COULD Be Used (Future)**

```
Query: "Find mild Alzheimer cases with atrophy"
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ MAMBA SSM MODEL                                              │
│                                                              │
│ Input Prompt:                                               │
│ """                                                          │
│ You are a medical image retrieval assistant.                │
│                                                              │
│ Query: "Find mild Alzheimer cases with atrophy"            │
│                                                              │
│ Analyze and respond in JSON:                                │
│ {                                                            │
│   "intent": "local_search|global_search|hybrid_search",    │
│   "disease": "alzheimer|brain_tumor|parkinson|ms",          │
│   "severity": "mild|moderate|severe|none",                  │
│   "features": ["atrophy", ...],                             │
│   "comparison": true/false                                  │
│ }                                                            │
│ """                                                          │
│                                                              │
│ Mamba Processing:                                           │
│   → State-space transformations                             │
│   → Contextual understanding                                │
│   → Intent classification                                   │
│                                                              │
│ Output:                                                     │
│ {                                                            │
│   "intent": "local_search",                                 │
│   "disease": "alzheimer",                                   │
│   "severity": "mild",                                       │
│   "features": ["atrophy", "hippocampus"],                   │
│   "comparison": false                                       │
│ }                                                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ QUERY RESULT OBJECT                                          │
│                                                              │
│ QueryResult(                                                │
│     raw_query="Find mild Alzheimer cases with atrophy",     │
│     intent="local_search",                                  │
│     entities={                                              │
│         'disease': 'alzheimer',                             │
│         'severity': 'mild',                                 │
│         'features': ['atrophy']                             │
│     },                                                       │
│     embedding=np.array([0.23, -0.12, ...]),  # From CLIP   │
│     confidence=0.95                                         │
│ )                                                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
             Sent to EnhancedGraphRAGRetriever
             for actual image retrieval
```

---

## 🔑 **KEY COMPONENTS & THEIR ROLES**

### **1. CLIP Model** (`src/clip_embeddings.py`)

**Role**: Convert images and text to 512-dim embeddings

```python
# Images → Embeddings
image = load("alzheimer_001.jpg")
image_emb = CLIP.get_image_features(image)  # 512-dim vector

# Text → Embeddings
query = "Find mild Alzheimer cases"
query_emb = CLIP.get_text_features(query)   # 512-dim vector

# Same embedding space → Can compare!
similarity = cosine_similarity(image_emb, query_emb)
```

**Storage**: Embeddings cached in `embeddings_cache/*.pkl`

---

### **2. SSM/Mamba** (`src/ssm.py`)

**Role**: Understand query intent and select search strategy

```python
query = "Find mild Alzheimer cases"
  │
  ▼
SSM analyzes:
  - Intent: "find" → local_search
  - Entities: "mild", "Alzheimer"
  - Mode: specific retrieval
  │
  ▼
Returns: QueryResult(intent="local_search", entities={...})
```

**Storage**: Model loaded in memory (130M parameters)

---

### **3. Community Detector** (`src/community_detection.py`)

**Role**: Group similar images into hierarchical communities

```python
# Runs ONCE during graph building
embeddings = [img_001_emb, img_002_emb, ..., img_400_emb]
  │
  ▼
Level 0: Group by disease
  → 4 communities (alzheimer, tumor, parkinson, ms)
  │
  ▼
Level 1: Within each disease, cluster by visual similarity
  → ~16 communities (mild, moderate, severe, subtypes)
  │
  ▼
Level 2: Fine-grained within each cluster
  → ~24 communities (specific patterns)
  │
  ▼
Total: 44 communities stored in graph
```

**Storage**: Community assignments stored in graph nodes

---

### **4. Knowledge Graph** (`src/graphRAG.py`)

**Role**: Store all relationships and enable graph traversal

```python
graph = {
    'nodes': {
        'img_001': {
            'embedding': [...],
            'communities': ['L0_C0', 'L1_C2', 'L2_C5'],
            'metadata': {...}
        }
    },
    'edges': [
        ('img_001', 'img_002', {'similarity': 0.87}),
        ('img_001', 'img_045', {'similarity': 0.76}),
        ...
    ]
}
```

**Storage**: NetworkX graph object in memory

---

### **5. Enhanced Retriever** (`src/enhanced_graphrag.py`)

**Role**: Orchestrate retrieval using communities and embeddings

```python
def retrieve(query):
    # 1. Process query with SSM
    query_result = ssm_processor.process_query(query)
    
    # 2. Select strategy
    if query_result.intent == "local":
        results = local_search(query_result.embedding)
    elif query_result.intent == "global":
        results = global_search(query_result)
    else:
        results = hybrid_search(query_result)
    
    # 3. Rank and return
    return top_k_results
```

**Storage**: All data structures in memory during runtime

---

## 📊 **DATA FLOW DIAGRAM**

```
┌──────────────────────┐
│   Raw Images (400)   │  Stored on disk
│   - alzheimer: 100   │  balanced_data/
│   - tumor: 100       │
│   - parkinson: 100   │
│   - ms: 100          │
└──────────┬───────────┘
           │
           │ Load & Process
           ▼
┌──────────────────────┐
│  CLIP Embedding      │  Extract features
│  Extractor           │  512-dim vectors
└──────────┬───────────┘
           │
           │ Cache to disk
           ▼
┌──────────────────────┐
│ Embeddings Cache     │  Stored as .pkl
│ - alzheimer.pkl      │  embeddings_cache/
│ - tumor.pkl          │  
│ - parkinson.pkl      │  (Reload on startup)
│ - ms.pkl             │
└──────────┬───────────┘
           │
           │ Load into memory
           ▼
┌──────────────────────────────────────────────┐
│        IN-MEMORY DATA STRUCTURES             │
│                                              │
│  ┌────────────────┐  ┌──────────────────┐  │
│  │ Image          │  │ Knowledge        │  │
│  │ Embeddings     │──│ Graph            │  │
│  │ (400 × 512)    │  │ (400 nodes,      │  │
│  └────────────────┘  │  75K edges)      │  │
│                      └──────────────────┘  │
│  ┌────────────────┐  ┌──────────────────┐  │
│  │ Communities    │  │ Metadata         │  │
│  │ (44 groups)    │  │ Index            │  │
│  └────────────────┘  └──────────────────┘  │
└──────────────────┬───────────────────────────┘
                   │
                   │ Query comes in
                   ▼
┌──────────────────────────────────────────────┐
│  QUERY PROCESSING                             │
│                                              │
│  User Query → SSM → Search Strategy          │
│             → Retrieval → Ranking            │
└──────────────────┬───────────────────────────┘
                   │
                   │ Results
                   ▼
┌──────────────────────────────────────────────┐
│  RETURN                                       │
│  - Top-K images                              │
│  - Similarity scores                         │
│  - Community explanations                    │
│  - Confidence values                         │
└──────────────────────────────────────────────┘
```

---

## ⏱️ **TIMING BREAKDOWN**

**Total Query Time: ~71ms**

```
Query: "Find mild Alzheimer cases"
  │
  ├─ SSM Processing: ~5ms
  │   └─ Intent detection, entity extraction
  │
  ├─ CLIP Text Embedding: ~10ms
  │   └─ Convert query to 512-dim vector
  │
  ├─ Similarity Computation: ~40ms
  │   └─ Compare query vs 400 image embeddings
  │
  ├─ Ranking & Filtering: ~5ms
  │   └─ Sort results, get top-K
  │
  └─ Explanation Generation: ~11ms
      └─ Load metadata, community paths
  │
  ▼
Total: ~71ms
```

---

## 💾 **MEMORY USAGE**

```
When System is Running:
  
  ├─ Image Embeddings: 400 × 512 × 4 bytes = ~800 KB
  ├─ Graph Structure: ~10 MB
  ├─ Communities: ~2 MB
  ├─ Metadata: ~1 MB
  ├─ CLIP Model: ~600 MB (loaded in memory)
  └─ SSM Model: ~500 MB (if using Mamba)
  
  Total: ~1.1 GB RAM
```

---

## 🎯 **SUMMARY**

### **Where You Are:**

✅ **400 images** organized in 4 disease datasets  
✅ **44 hierarchical communities** (3 levels)  
✅ **75,625 edges** in knowledge graph  
✅ **CLIP embeddings** (512-dim) for all images  
✅ **Multi-strategy retrieval** (Global/Local/Hybrid)  
✅ **72.4% P@5** performance  

### **How Storage Works:**

📁 **Disk**: Raw images + cached embeddings (.pkl files)  
🧠 **Memory**: Graph, communities, embeddings loaded at runtime  
⚡ **Speed**: Cache avoids re-computing embeddings  

### **How Retrieval Works:**

1. **Query** → SSM processes intent
2. **Embedding** → CLIP converts query to 512-dim vector
3. **Search** → Multi-strategy (Global/Local/Hybrid)
4. **Rank** → Sort by similarity, enrich with context
5. **Return** → Top-K images with explanations

### **How Mamba Communicates:**

🔄 **Currently**: Rule-based intent detection (fast, simple)  
🔮 **Future**: Mamba LLM for advanced query understanding  
🔗 **Interface**: SSMQueryProcessor → QueryResult → Retriever  

---

**🎉 Your system is a complete end-to-end pipeline from raw images to interpretable retrieval results!**


