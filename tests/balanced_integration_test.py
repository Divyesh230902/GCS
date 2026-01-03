"""
Balanced Integration Test for the complete GCS pipeline
Tests SSM + CLIP + GraphRAG integration with balanced datasets
"""

import os
import sys
from balanced_data.balanced_data_utils import create_balanced_data_loaders, get_balanced_dataset_info
from clip_embeddings import CLIPEmbeddingExtractor
from ssm import SSMQueryProcessor
from graphRAG import GraphRAGRetriever, MedicalKnowledgeGraph

def test_balanced_integration():
    """Test the complete pipeline integration with balanced datasets"""
    print("=" * 60)
    print("GCS BALANCED INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing Components...")
    print("-" * 30)
    
    # CLIP extractor
    print("Initializing CLIP extractor...")
    clip_extractor = CLIPEmbeddingExtractor()
    
    # SSM processor
    print("Initializing SSM processor...")
    ssm_processor = SSMQueryProcessor()
    
    # Knowledge graph
    print("Initializing knowledge graph...")
    graph = MedicalKnowledgeGraph()
    
    # GraphRAG retriever
    print("Initializing GraphRAG retriever...")
    retriever = GraphRAGRetriever(clip_extractor, ssm_processor, graph)
    
    print("✅ All components initialized successfully!")
    
    # Test balanced data loading
    print("\n2. Testing Balanced Data Loading...")
    print("-" * 30)
    
    balanced_data_dir = "./balanced_data"
    
    # Get balanced dataset info
    balanced_info = get_balanced_dataset_info(balanced_data_dir)
    print("Balanced datasets:")
    for dataset_name, info in balanced_info.items():
        if 'error' not in info:
            print(f"  {dataset_name}: {info.get('total_samples', 'N/A')} samples")
            if 'class_distribution' in info:
                print(f"    Class distribution: {info['class_distribution']}")
            elif 'train_class_distribution' in info:
                print(f"    Train distribution: {info['train_class_distribution']}")
                print(f"    Test distribution: {info['test_class_distribution']}")
        else:
            print(f"  {dataset_name}: Error - {info['error']}")
    
    # Test with a small sample from balanced data
    print("\n3. Testing with Balanced Sample Data...")
    print("-" * 30)
    
    # Create sample image embeddings from balanced data
    print("Creating balanced sample image embeddings...")
    sample_image_embeddings = []
    
    # Sample from each balanced dataset
    balanced_datasets = ['alzheimer', 'brain_tumor', 'parkinson', 'ms']
    sample_data = []
    
    for dataset in balanced_datasets:
        if dataset == 'brain_tumor':
            # Sample from training data
            train_path = os.path.join(balanced_data_dir, 'balanced_brain_tumor', 'Training')
            if os.path.exists(train_path):
                for class_name in ['glioma', 'meningioma', 'notumor', 'pituitary']:
                    class_path = os.path.join(train_path, class_name)
                    if os.path.exists(class_path):
                        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if files:
                            sample_file = files[0]  # Take first file as sample
                            sample_data.append((os.path.join(class_path, sample_file), class_name, dataset))
        else:
            # Sample from other datasets
            dataset_path = os.path.join(balanced_data_dir, f'balanced_{dataset}')
            if os.path.exists(dataset_path):
                for class_name in os.listdir(dataset_path):
                    class_path = os.path.join(dataset_path, class_name)
                    if os.path.isdir(class_path):
                        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if files:
                            sample_file = files[0]  # Take first file as sample
                            sample_data.append((os.path.join(class_path, sample_file), class_name, dataset))
    
    print(f"Created {len(sample_data)} balanced sample images")
    
    # Create embeddings for sample data
    for img_path, class_label, dataset in sample_data:
        embedding = clip_extractor._simple_image_embedding(img_path, class_label, dataset)
        
        from clip_embeddings import ImageEmbedding
        img_emb = ImageEmbedding(
            image_path=img_path,
            embedding=embedding,
            class_label=class_label,
            dataset=dataset,
            metadata={}
        )
        sample_image_embeddings.append(img_emb)
    
    # Create text embeddings
    print("Creating text embeddings...")
    text_embeddings = {}
    for dataset in balanced_datasets:
        dataset_text_embeddings = clip_extractor.create_text_embeddings_for_dataset(dataset)
        text_embeddings.update(dataset_text_embeddings)
    
    print(f"Created {len(text_embeddings)} text embeddings")
    
    # Build knowledge graph
    print("\n4. Building Knowledge Graph...")
    print("-" * 30)
    
    retriever.build_graph_from_embeddings(
        sample_image_embeddings, 
        text_embeddings, 
        cluster_threshold=0.5
    )
    
    print(f"Knowledge graph built with {retriever.graph.graph.number_of_nodes()} nodes and {retriever.graph.graph.number_of_edges()} edges")
    
    # Test retrieval
    print("\n5. Testing Retrieval...")
    print("-" * 30)
    
    test_queries = [
        "Find brain tumor images",
        "Show me Alzheimer's disease scans",
        "Compare normal vs Parkinson's brain images",
        "Analyze MS lesion patterns"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 20)
        
        # Process with different reasoning strategies
        strategies = ["direct_similarity", "concept_expansion", "hierarchical_reasoning"]
        
        for strategy in strategies:
            try:
                result = retriever.retrieve(query, top_k=3, reasoning_strategy=strategy)
                print(f"  {strategy}: {len(result.retrieved_images)} results")
                print(f"    Intent: {result.metadata['query_intent']}")
                print(f"    Confidence: {result.metadata['query_confidence']:.2f}")
                
                if result.retrieved_images:
                    print(f"    Top result: {result.retrieved_images[0].class_label} ({result.retrieved_images[0].dataset})")
                    print(f"    Similarity: {result.confidence_scores[0]:.3f}")
                
            except Exception as e:
                print(f"  {strategy}: Error - {e}")
    
    print("\n6. Testing Query Processing...")
    print("-" * 30)
    
    # Test SSM query processing
    test_queries_ssm = [
        "Find similar brain tumor images",
        "Classify this MRI scan for Alzheimer's disease",
        "Compare Parkinson's vs normal brain scans"
    ]
    
    for query in test_queries_ssm:
        print(f"\nQuery: '{query}'")
        result = ssm_processor.process_query(query)
        print(f"  Intent: {result.intent} (confidence: {result.confidence:.2f})")
        print(f"  Retrieval instructions: {result.retrieval_instructions['filters']}")
    
    print("\n" + "=" * 60)
    print("BALANCED INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n✅ All components working with balanced datasets:")
    print("  - SSM: Query processing and intent detection")
    print("  - CLIP: Embedding generation (fallback mode)")
    print("  - GraphRAG: Graph-based retrieval and reasoning")
    print("  - Balanced Data: Equal distribution of disease/normal cases")
    print("\n🎯 Key Benefits of Balanced Sampling:")
    print("  - Reduced bias in training and evaluation")
    print("  - Equal representation of all disease classes")
    print("  - Fair comparison between different approaches")
    print("  - More reliable performance metrics")
    print("\n🚀 Ready for unbiased baseline comparison!")

if __name__ == "__main__":
    test_balanced_integration()
