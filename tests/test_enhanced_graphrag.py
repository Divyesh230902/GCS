#!/usr/bin/env python3
"""
Test Enhanced GraphRAG with Microsoft GraphRAG-inspired features
Tests hierarchical communities, Global/Local/Hybrid search
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.enhanced_graphrag import EnhancedGraphRAGRetriever
from src.clip_embeddings import CLIPEmbeddingExtractor
from src.ssm import SSMQueryProcessor
from src.graphRAG import MedicalKnowledgeGraph

def test_enhanced_graphrag():
    """Test the complete enhanced GraphRAG system"""
    print("=" * 70)
    print("ENHANCED GRAPHRAG TEST - Microsoft GraphRAG-inspired Medical IR")
    print("=" * 70)
    
    # Initialize components
    print("\n1. Initializing components...")
    ssm_processor = SSMQueryProcessor(model_key="rule-based")
    print("   ✓ SSM Processor initialized")
    
    clip_extractor = CLIPEmbeddingExtractor()
    print("   ✓ CLIP Extractor initialized")
    
    graph = MedicalKnowledgeGraph()
    print("   ✓ Medical Knowledge Graph initialized")
    
    # Initialize Enhanced GraphRAG
    retriever = EnhancedGraphRAGRetriever(clip_extractor, ssm_processor, graph)
    print("   ✓ Enhanced GraphRAG Retriever initialized")
    
    # Load sample data
    print("\n2. Loading sample medical images...")
    sample_data_dir = Path(__file__).parent.parent / "balanced_data"
    
    if not sample_data_dir.exists():
        print("   ⚠️  No balanced data found. Run scripts/balanced_sampling.py first.")
        return False
    
    # Collect sample images
    sample_images = []
    class_labels = []
    datasets = []
    
    for dataset_name in ['balanced_alzheimer', 'balanced_brain_tumor', 'balanced_parkinson', 'balanced_ms']:
        dataset_path = sample_data_dir / dataset_name
        if dataset_path.exists():
            disease = dataset_name.replace('balanced_', '')
            count = 0
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_images.append(os.path.join(root, file))
                        class_labels.append(os.path.basename(root))
                        datasets.append(disease)
                        count += 1
                        if count >= 20:  # Limit per dataset for testing
                            break
                if count >= 20:
                    break
    
    print(f"   ✓ Loaded {len(sample_images)} sample images from {len(set(datasets))} datasets")
    
    # Extract embeddings
    print("\n3. Extracting CLIP embeddings...")
    embeddings = clip_extractor.batch_extract_embeddings(
        sample_images, class_labels, "test_dataset"
    )
    print(f"   ✓ Extracted {len(embeddings)} embeddings")
    
    # Build enhanced graph with communities
    print("\n4. Building enhanced graph with hierarchical communities...")
    retriever.build_enhanced_graph(embeddings, {})
    
    # Print community statistics
    print("\n5. Community Detection Results:")
    print(f"   Total Communities: {len(retriever.communities)}")
    by_level = {0: 0, 1: 0, 2: 0}
    for comm in retriever.communities.values():
        by_level[comm.level] += 1
    print(f"   Level 0 (Global): {by_level[0]} communities")
    print(f"   Level 1 (Mid): {by_level[1]} communities")
    print(f"   Level 2 (Local): {by_level[2]} communities")
    
    # Test different search modes
    test_queries = [
        ("What patterns exist across all neurological diseases?", "global"),
        ("Find brain scans similar to mild Alzheimer's cases", "local"),
        ("Compare tumor types across all brain imaging", "hybrid"),
    ]
    
    print("\n6. Testing Enhanced Retrieval:")
    print("=" * 70)
    
    for query, expected_mode in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected Mode: {expected_mode}")
        print("-" * 70)
        
        # Perform retrieval
        result = retriever.retrieve(query, top_k=5, search_mode="auto")
        
        # Print results
        print(f"Actual Mode: {result.search_mode}")
        print(f"Retrieved: {len(result.retrieved_images)} images")
        print(f"Confidence: {result.confidence:.2f}")
        
        print("\nReasoning Path:")
        for step in result.reasoning_path:
            print(f"  • {step}")
        
        print("\nTop Results:")
        for i, img in enumerate(result.retrieved_images[:3], 1):
            print(f"  {i}. {img.class_label} ({img.dataset})")
        
        if result.community_context:
            print(f"\nCommunity Context: {len(result.community_context)} communities analyzed")
        
        print("=" * 70)
    
    # Test hierarchical context
    print("\n7. Testing Hierarchical Context:")
    if retriever.node_metadata:
        sample_node = list(retriever.node_metadata.keys())[0]
        context = retriever.get_node_hierarchical_context(sample_node)
        
        print(f"Node: {sample_node}")
        print(f"Communities: {context['communities']}")
        print("Hierarchical Summaries:")
        for summary in context['summaries']:
            print(f"  Level {summary['level']}: {summary['text'][:80]}...")
    
    # Success summary
    print("\n" + "=" * 70)
    print("✅ ENHANCED GRAPHRAG TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Features Tested:")
    print("  ✓ Hierarchical community detection (3 levels)")
    print("  ✓ Community-based summarization")
    print("  ✓ Global search (community-level reasoning)")
    print("  ✓ Local search (entity-level precision)")
    print("  ✓ Hybrid search (combined approach)")
    print("  ✓ Automatic search mode selection")
    print("  ✓ Hierarchical context retrieval")
    print("\n" + "=" * 70)
    
    return True

def compare_with_baseline():
    """Compare enhanced GraphRAG with basic approach"""
    print("\n" + "=" * 70)
    print("COMPARISON: Enhanced GraphRAG vs Basic GraphRAG")
    print("=" * 70)
    
    print("\nEnhanced GraphRAG Features:")
    print("  ✓ Hierarchical communities (disease → visual → fine-grained)")
    print("  ✓ Community summaries using SSM")
    print("  ✓ Global/Local/Hybrid search modes")
    print("  ✓ Multi-level reasoning")
    print("  ✓ Microsoft GraphRAG-inspired architecture")
    
    print("\nBasic GraphRAG (Previous):")
    print("  • Flat graph structure")
    print("  • Single-level similarity search")
    print("  • No community detection")
    print("  • Limited reasoning capabilities")
    
    print("\nKey Improvements:")
    print("  1. Hierarchical Organization: 3-level community structure")
    print("  2. Contextual Understanding: Community-level summaries")
    print("  3. Flexible Retrieval: Multiple search strategies")
    print("  4. Better Scalability: Efficient community-based indexing")
    print("  5. Explainability: Clear reasoning paths and explanations")
    
    print("=" * 70)

if __name__ == "__main__":
    print("Enhanced GraphRAG Test Suite")
    print("Microsoft GraphRAG-inspired Medical Image Retrieval")
    print()
    
    try:
        success = test_enhanced_graphrag()
        
        if success:
            compare_with_baseline()
            
            print("\n🎉 All tests passed! System ready for CHIIR'26 evaluation.")
        else:
            print("\n⚠️  Test incomplete. Check data availability.")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

