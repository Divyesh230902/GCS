# GCS: Graph-based Classification System for Medical Image Retrieval

## Novel Framework Architecture

This project implements a novel information retrieval framework that combines:

1. **SSM (State Space Model)** - Query handling and data retrieval orchestration
2. **CLIP Embeddings** - Multimodal embedding generation for medical images  
3. **GraphRAG** - Graph-based retrieval and reasoning over medical data

## Datasets
- Alzheimer's Disease (4 classes, ~86K images)
- Brain Tumor Classification (4 classes, ~6K images) 
- Parkinson's Disease (2 classes, 900 images)
- Multiple Sclerosis (2 classes, 420 images)

## Comparison
- **Baseline**: Traditional multimodal RAG approaches
- **Novel**: SSM + CLIP + GraphRAG pipeline