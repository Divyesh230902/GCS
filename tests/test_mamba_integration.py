#!/usr/bin/env python3
"""
Test Mamba SSM Integration with GCS
Tests the complete pipeline with Mamba state-space model
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.ssm import SSMQueryProcessor
from src.clip_embeddings import CLIPEmbeddingExtractor
from src.graphRAG import GraphRAGRetriever
from src.model_config import get_available_models, check_huggingface_auth, setup_huggingface_auth

def test_mamba_models():
    """Test different Mamba model configurations"""
    print("=" * 60)
    print("TESTING MAMBA SSM MODELS")
    print("=" * 60)
    
    # Check authentication status
    has_auth = check_huggingface_auth()
    print(f"Hugging Face Authentication: {'✅ Available' if has_auth else '❌ Not Available'}")
    
    if not has_auth:
        print("\nTo use Mamba models, you need Hugging Face authentication:")
        setup_huggingface_auth()
        print()
    
    # Test different model configurations
    model_configs = ["mamba-1.4b", "mamba-130m", "gpt2", "rule-based"]
    
    for model_key in model_configs:
        print(f"\n{'='*40}")
        print(f"Testing Model: {model_key}")
        print(f"{'='*40}")
        
        try:
            # Initialize SSM processor
            ssm_processor = SSMQueryProcessor(model_key=model_key)
            
            # Test queries
            test_queries = [
                "Find brain tumor images with glioma",
                "Classify this MRI for Alzheimer's disease",
                "Compare normal vs Parkinson's brain scans",
                "Analyze MS lesion patterns"
            ]
            
            for query in test_queries:
                print(f"\nQuery: {query}")
                result = ssm_processor.process_query(query)
                print(f"Intent: {result.intent} (confidence: {result.confidence})")
                print(f"Instructions: {result.retrieval_instructions}")
                
                # Test embedding generation
                try:
                    embedding = ssm_processor.generate_embedding_query(query)
                    print(f"Embedding shape: {embedding.shape}")
                except Exception as e:
                    print(f"Embedding generation failed: {e}")
            
        except Exception as e:
            print(f"Error testing {model_key}: {e}")

def test_complete_pipeline():
    """Test the complete GCS pipeline with Mamba"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE GCS PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize components
        print("Initializing GCS components...")
        
        # SSM Query Processor (with Mamba)
        ssm_processor = SSMQueryProcessor(model_key="mamba-1.4b")
        print("✅ SSM Processor initialized")
        
        # CLIP Embedding Extractor
        clip_extractor = CLIPEmbeddingExtractor()
        print("✅ CLIP Extractor initialized")
        
        # Medical Knowledge Graph
        from src.graphRAG import MedicalKnowledgeGraph
        graph = MedicalKnowledgeGraph()
        
        # GraphRAG Retriever
        retriever = GraphRAGRetriever(clip_extractor, ssm_processor, graph)
        print("✅ GraphRAG Retriever initialized")
        
        # Test query processing
        query = "Find similar brain tumor images with glioma"
        print(f"\nProcessing query: {query}")
        
        # Process query with SSM
        query_result = ssm_processor.process_query(query)
        print(f"SSM Intent: {query_result.intent}")
        print(f"SSM Instructions: {query_result.retrieval_instructions}")
        
        # Generate query embedding
        query_embedding = ssm_processor.generate_embedding_query(query)
        print(f"Query embedding shape: {query_embedding.shape}")
        
        # Test with sample data (if available)
        sample_data_dir = "./balanced_data"
        if os.path.exists(sample_data_dir):
            print(f"\nTesting with balanced data from: {sample_data_dir}")
            
            # Load some sample images
            sample_images = []
            for root, dirs, files in os.walk(sample_data_dir):
                for file in files[:5]:  # Take first 5 images
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_images.append(os.path.join(root, file))
                        if len(sample_images) >= 5:
                            break
                if len(sample_images) >= 5:
                    break
            
            if sample_images:
                print(f"Found {len(sample_images)} sample images")
                
                # Extract embeddings
                class_labels = [os.path.basename(os.path.dirname(img)) for img in sample_images]
                dataset = "test"
                embeddings = clip_extractor.batch_extract_embeddings(sample_images, class_labels, dataset)
                print(f"Extracted {len(embeddings)} image embeddings")
                
                # Build graph (text embeddings will be generated internally)
                retriever.build_graph_from_embeddings(embeddings, {})
                print(f"Graph built with {retriever.graph.graph.number_of_nodes()} nodes")
                
                # Test retrieval
                results = retriever.retrieve(query, top_k=3)
                print(f"Retrieved {len(results)} results")
                
                for i, result in enumerate(results):
                    print(f"  {i+1}. {result['image_path']} (similarity: {result['similarity']:.3f})")
            else:
                print("No sample images found for testing")
        else:
            print("Balanced data directory not found. Run balanced_sampling.py first.")
        
        print("\n✅ Complete pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("MAMBA SSM INTEGRATION TEST")
    print("=" * 60)
    
    # Test available models
    available_models = get_available_models()
    print("Available models:")
    for key, desc in available_models.items():
        print(f"  {key}: {desc}")
    
    # Test Mamba models
    test_mamba_models()
    
    # Test complete pipeline
    test_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
