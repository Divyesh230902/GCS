# 📝 Methodology Section - Recommendations for CHIIR'26 Paper

**Based on your proposed sections and our implementation**

---

## 🎯 **YOUR PROPOSED SECTIONS:**

1. Multimodal Representation Learning
2. Medical Knowledge Graph for Retrieval
3. Hierarchical Community Detection
4. Global & Local Retrieval
5. State-Space Model

---

## ✅ **MY RECOMMENDATION: REVISED FLOW**

### **Suggested Order & Names:**

1. **Multimodal Representation Learning** ✅ (Keep)
2. **Medical Knowledge Graph Construction** ✏️ (Minor rename)
3. **Hierarchical Community Detection** ✅ (Keep)
4. **Multi-Strategy Retrieval Framework** ✏️ (Rename + reorder)
5. **Query Processing with State-Space Models** ✏️ (Rename)

---

## 📊 **DETAILED SECTION BREAKDOWN**

### **SECTION 1: Multimodal Representation Learning** ✅

**Status**: PERFECT - Keep as is!

**Why it's good**:
- Sets foundation (CLIP embeddings)
- Explains how images and queries are represented
- Standard ML paper structure (features first)

**What to cover**:
```
3.1 Multimodal Representation Learning
  3.1.1 CLIP Embedding Extraction
    - Image encoder: ViT-B/32 → 512-dim vectors
    - Text encoder: Transformer → 512-dim vectors
    - Shared embedding space for similarity
  
  3.1.2 Medical Image Preprocessing
    - RGB conversion
    - Normalization
    - Batch processing
  
  3.1.3 Mathematical Formulation
    - x_i = CLIP_image(I_i) ∈ ℝ^512
    - q = CLIP_text(Q) ∈ ℝ^512
    - sim(x_i, q) = (x_i · q) / (||x_i|| ||q||)
```

**Key equations to include**:
$$\mathbf{x}_i = f_{\text{CLIP}}^{\text{img}}(I_i) \in \mathbb{R}^{512}$$
$$\mathbf{q} = f_{\text{CLIP}}^{\text{text}}(Q) \in \mathbb{R}^{512}$$
$$\text{sim}(\mathbf{x}_i, \mathbf{q}) = \frac{\mathbf{x}_i \cdot \mathbf{q}}{\|\mathbf{x}_i\| \|\mathbf{q}\|}$$

---

### **SECTION 2: Medical Knowledge Graph Construction** ✏️

**Status**: Rename from "Medical Knowledge Graph for Retrieval"

**Why rename**:
- "Construction" is clearer (you BUILD the graph)
- "for Retrieval" is redundant (everything is for retrieval)
- Separates building vs. using the graph

**What to cover**:
```
3.2 Medical Knowledge Graph Construction
  3.2.1 Graph Schema
    - Nodes: Images with metadata (disease, class, embeddings)
    - Edges: Cosine similarity > threshold (0.7)
    - Properties: Dense graph (95% connectivity)
  
  3.2.2 Similarity-Based Edge Construction
    - Compute pairwise similarities
    - Threshold filtering
    - Edge weighting
  
  3.2.3 Graph Statistics
    - 400 nodes (images)
    - 75,625 edges
    - Average degree: 378
```

**Key formulation**:
$$G = (V, E)$$
$$V = \{v_1, v_2, ..., v_n\}, \quad n = 400$$
$$E = \{(v_i, v_j, w_{ij}) \mid \text{sim}(\mathbf{x}_i, \mathbf{x}_j) > \tau\}$$
$$\tau = 0.7 \quad \text{(similarity threshold)}$$

**Figure suggestion**:
- Graph visualization showing dense connectivity
- Example node with metadata
- Edge weight distribution

---

### **SECTION 3: Hierarchical Community Detection** ✅

**Status**: PERFECT - Keep as is!

**Why it's good**:
- Core novelty of your work
- Natural flow after graph construction
- Clear hierarchical structure

**What to cover**:
```
3.3 Hierarchical Community Detection
  3.3.1 Three-Level Hierarchy
    - Level 0: Disease-type communities (4 communities)
    - Level 1: Visual similarity clusters (~16 communities)
    - Level 2: Fine-grained class communities (~24 communities)
  
  3.3.2 Agglomerative Clustering Algorithm
    - Ward linkage criterion
    - Adaptive cluster count: k = min(max(2, ⌊|V|/10⌋), 5)
    - Bottom-up merging
  
  3.3.3 Community Summarization
    - SSM-generated textual summaries
    - Centroid computation
    - Parent-child relationships
```

**Key algorithm** (use Algorithm box):
```
Algorithm 1: Hierarchical Community Detection

Input: G = (V, E), embeddings X, metadata M, levels L_max = 3
Output: Communities C = {C_0, C_1, C_2}

1: // Level 0: Disease-based grouping
2: C_0 ← GroupByDisease(V, M)
3: 
4: // Level 1: Visual clustering within each disease
5: for each C ∈ C_0 do
6:    X_C ← {x_i | v_i ∈ C}
7:    k ← min(max(2, ⌊|C|/10⌋), 5)
8:    C_1 ← AgglomerativeClustering(X_C, k, linkage='ward')
9: end for
10:
11: // Level 2: Class-based refinement
12: for each C ∈ C_1 do
13:    C_2 ← GroupByClassLabel(C, M)
14: end for
15:
16: return {C_0, C_1, C_2}
```

**Mathematical formulation**:
$$\Delta(C_a, C_b) = \frac{|C_a| \cdot |C_b|}{|C_a| + |C_b|} \|\boldsymbol{\mu}_a - \boldsymbol{\mu}_b\|^2$$
(Ward linkage distance)

$$\boldsymbol{\mu}_c = \frac{1}{|C|} \sum_{v_i \in C} \mathbf{x}_i$$
(Community centroid)

**Figure suggestion**:
- Hierarchical tree diagram (Level 0 → L1 → L2)
- Community size distribution
- Example community with sample images

---

### **SECTION 4: Multi-Strategy Retrieval Framework** ✏️

**Status**: Rename from "Global & Local Retrieval"

**Why rename**:
- "Multi-Strategy" emphasizes your novelty (3 modes)
- "Framework" indicates systematic approach
- More comprehensive than just "Global & Local"

**What to cover**:
```
3.4 Multi-Strategy Retrieval Framework
  3.4.1 Search Mode Overview
    - Global Search: Community-based (broad queries)
    - Local Search: Entity-based (specific queries)
    - Hybrid Search: Combined approach
  
  3.4.2 Global Search (Community-Level)
    - Query → Community summary matching
    - Aggregate results from top-K communities
    - Best for: "Show all...", "Compare patterns..."
  
  3.4.3 Local Search (Entity-Level)
    - Query → Direct embedding similarity
    - Return top-K most similar images
    - Best for: "Find specific...", "Retrieve cases..."
  
  3.4.4 Hybrid Search (Combined)
    - Weighted combination: α·global + (1-α)·local
    - Balances coverage and precision
    - Best for: Classification, ambiguous queries
  
  3.4.5 Automatic Mode Selection
    - Intent-based routing (from SSM)
    - Keyword patterns
    - Query complexity analysis
```

**Key formulations**:

**Global Search**:
$$\text{Score}_{\text{global}}(I_i, Q) = \max_{C \in \mathcal{C}} \text{sim}(\text{summary}(C), Q) \cdot \mathbb{1}[I_i \in C]$$

**Local Search**:
$$\text{Score}_{\text{local}}(I_i, Q) = \text{sim}(\mathbf{x}_i, \mathbf{q})$$

**Hybrid Search**:
$$\text{Score}_{\text{hybrid}}(I_i, Q) = \alpha \cdot \text{Score}_{\text{global}}(I_i, Q) + (1-\alpha) \cdot \text{Score}_{\text{local}}(I_i, Q)$$
where $\alpha = 0.5$ (default)

**Figure suggestion**:
- Flowchart showing mode selection
- Side-by-side comparison of 3 strategies
- Example queries for each mode

---

### **SECTION 5: Query Processing with State-Space Models** ✏️

**Status**: Rename from "State-Space Model"

**Why rename**:
- More descriptive (what SSM DOES, not just what it IS)
- "Query Processing" clarifies its role
- Better parallel with other section names

**What to cover**:
```
3.5 Query Processing with State-Space Models
  3.5.1 Query Understanding Pipeline
    - Natural language input
    - Intent detection (local/global/hybrid)
    - Entity extraction (disease, severity, features)
  
  3.5.2 Rule-Based Intent Detection (Current)
    - Keyword pattern matching
    - Heuristic classification
    - Efficient (5ms overhead)
  
  3.5.3 SSM-Enhanced Processing (Optional)
    - Mamba architecture for context understanding
    - Structured output generation
    - Future extension
  
  3.5.4 Query Embedding Generation
    - CLIP text encoder
    - Integration with retrieval pipeline
```

**Key process flow**:
```
Query Q → Intent Detection → Search Mode m
                           → Entity Extraction → Entities E
                           → CLIP Encoding → Query embedding q
```

**Formulation**:
$$m = f_{\text{intent}}(Q) \in \{\text{global}, \text{local}, \text{hybrid}\}$$
$$\mathcal{E} = f_{\text{extract}}(Q) = \{\text{disease}, \text{severity}, \text{features}\}$$
$$\mathbf{q} = f_{\text{CLIP}}^{\text{text}}(Q)$$

**Table suggestion**:
| Query Type | Intent | Mode | Example |
|------------|--------|------|---------|
| Specific retrieval | retrieval | Local | "Find mild Alzheimer cases" |
| Broad exploration | analysis | Global | "Show all brain tumor patterns" |
| Comparison | comparison | Global | "Compare MS vs normal" |
| Classification | classification | Hybrid | "Classify this image" |

---

## 🔄 **ALTERNATIVE ORDERING SUGGESTION**

### **Option A: Your Original Order** (Good ✅)
1. Multimodal Representation Learning
2. Medical Knowledge Graph Construction
3. Hierarchical Community Detection
4. Multi-Strategy Retrieval Framework
5. Query Processing with SSM

**Pros**:
- Logical bottom-up (features → graph → communities → retrieval → query)
- Standard ML paper structure
- Easy to follow

---

### **Option B: Query-First Order** (Alternative 🔄)
1. **System Overview** (NEW)
2. **Query Processing with SSM** (MOVED UP)
3. Multimodal Representation Learning
4. Medical Knowledge Graph Construction
5. Hierarchical Community Detection
6. Multi-Strategy Retrieval Framework

**Pros**:
- Starts with user perspective (query → results)
- Motivates each component
- More engaging narrative

**Cons**:
- Less traditional
- Requires forward references

---

### **Option C: Hybrid Structure** (Recommended 🎯)

Keep your order but add subsections:

```
3. METHODOLOGY

3.1 System Overview (NEW - 1 page)
  - High-level architecture diagram
  - Data flow: Query → SSM → Graph → Communities → Retrieval → Results
  - Component interactions

3.2 Multimodal Representation Learning
  [Your content]

3.3 Medical Knowledge Graph Construction
  [Your content]

3.4 Hierarchical Community Detection
  [Your content]

3.5 Multi-Strategy Retrieval Framework
  [Your content]

3.6 Query Processing Pipeline
  [Your content]
```

**Why this is best**:
- Overview gives big picture first
- Then explains each component in detail
- Readers understand context before diving deep

---

## 📝 **NARRATION SUGGESTIONS**

### **For Section 1 (Multimodal Representation Learning):**

**Opening**:
> "We employ CLIP (Radford et al., 2021) as our multimodal foundation model to extract unified representations for both medical images and textual queries. CLIP's pre-training on 400M image-text pairs provides strong zero-shot generalization to medical imaging domains without fine-tuning."

**Transition to next section**:
> "These embeddings serve as the foundation for constructing a similarity-based knowledge graph, enabling structured retrieval across diverse medical conditions."

---

### **For Section 2 (Medical Knowledge Graph Construction):**

**Opening**:
> "To enable graph-based reasoning over medical images, we construct a dense similarity graph where nodes represent images and edges encode visual similarity relationships."

**Key insight**:
> "Unlike knowledge graphs built from symbolic relationships, our graph emerges naturally from the embedding space, requiring no manual annotation of relationships."

**Transition to next section**:
> "While the graph captures pairwise similarities, effective retrieval requires higher-level organizational structure. We address this through hierarchical community detection."

---

### **For Section 3 (Hierarchical Community Detection):**

**Opening**:
> "To organize medical images at multiple granularities, we introduce a three-level hierarchical community detection approach combining domain knowledge (disease types) with data-driven visual clustering."

**Novelty statement**:
> "Unlike prior work that uses modularity optimization (Louvain/Leiden), we employ agglomerative clustering on CLIP embeddings, which is better suited for dense similarity graphs (95% connectivity) and produces deterministic, reproducible communities."

**Transition to next section**:
> "The resulting hierarchical structure enables multiple retrieval strategies adapted to different query types, which we describe next."

---

### **For Section 4 (Multi-Strategy Retrieval Framework):**

**Opening**:
> "Medical image queries vary widely in scope and specificity—from broad analytical queries ('Show tumor patterns') to precise retrieval ('Find patient X's scan'). To address this diversity, we propose a multi-strategy retrieval framework with three complementary search modes."

**Comparison**:
> "While traditional retrieval systems use a single strategy (typically flat vector search), our framework automatically selects between global (community-based), local (entity-based), and hybrid approaches based on query intent."

**Transition to next section**:
> "Automatic mode selection requires understanding query intent and structure, which we achieve through query processing described in the next section."

---

### **For Section 5 (Query Processing with SSM):**

**Opening**:
> "To automatically route queries to appropriate search strategies, we develop a query processing pipeline that analyzes natural language queries to detect intent, extract medical entities, and generate embeddings."

**Implementation note**:
> "Our current implementation uses efficient rule-based intent detection with keyword pattern matching, achieving 5ms processing time. The architecture also supports enhancement with state-space models (Mamba) for more sophisticated query understanding."

**Closing**:
> "This query processing pipeline completes our end-to-end system, connecting user queries to retrieval results through the hierarchical graph structure."

---

## 🎯 **FINAL RECOMMENDATION**

### **✅ Use Your Original Sections with These Changes:**

1. ✅ **Multimodal Representation Learning** (KEEP)
2. ✏️ **Medical Knowledge Graph Construction** (Rename: remove "for Retrieval")
3. ✅ **Hierarchical Community Detection** (KEEP)
4. ✏️ **Multi-Strategy Retrieval Framework** (Rename: from "Global & Local Retrieval")
5. ✏️ **Query Processing with State-Space Models** (Rename: from "State-Space Model")

### **📌 Add at the beginning:**

**3.0 System Overview** (0.5-1 page)
- High-level architecture diagram
- Data flow overview
- Component relationships

---

## 📊 **SECTION FLOW DIAGRAM**

```
┌─────────────────────────────────────────────────────┐
│  3.0 System Overview (NEW)                          │
│  - Architecture diagram                             │
│  - Component relationships                          │
└──────────────────┬──────────────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
┌──────────────────┐      ┌──────────────────┐
│ 3.1 Multimodal   │      │ 3.5 Query        │
│ Representation   │◄────►│ Processing       │
└────────┬─────────┘      └──────────────────┘
         │                         │
         ▼                         │
┌──────────────────┐              │
│ 3.2 Knowledge    │              │
│ Graph            │              │
└────────┬─────────┘              │
         │                         │
         ▼                         │
┌──────────────────┐              │
│ 3.3 Hierarchical │              │
│ Communities      │              │
└────────┬─────────┘              │
         │                         │
         └─────────┬───────────────┘
                   ▼
         ┌──────────────────┐
         │ 3.4 Multi-       │
         │ Strategy         │
         │ Retrieval        │
         └──────────────────┘
```

---

## ✅ **SUMMARY OF CHANGES**

| Your Section | Recommended Name | Change |
|--------------|------------------|--------|
| (None) | System Overview | ➕ ADD |
| Multimodal Representation Learning | (Same) | ✅ Keep |
| Medical Knowledge Graph for Retrieval | Medical Knowledge Graph Construction | ✏️ Rename |
| Hierarchical Community Detection | (Same) | ✅ Keep |
| Global & Local Retrieval | Multi-Strategy Retrieval Framework | ✏️ Rename |
| State-Space Model | Query Processing with State-Space Models | ✏️ Rename |

---

## 🎓 **WHY THESE CHANGES HELP**

1. **"System Overview"** - Gives readers the big picture first
2. **"Construction"** - Clearer action (you BUILD the graph)
3. **"Multi-Strategy"** - Emphasizes novelty (3 modes, not just 2)
4. **"Framework"** - Indicates systematic, complete approach
5. **"Query Processing with..."** - Shows what SSM DOES, not just IS

---

**🎯 Bottom Line**: Your section structure is 95% perfect! Just add "System Overview" at the start and refine a few names to better reflect your contributions. The flow is logical and matches standard ML paper structure! 🎉


