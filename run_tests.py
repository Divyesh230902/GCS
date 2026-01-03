#!/usr/bin/env python3
"""
Run all GCS tests
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("RUNNING GCS TESTS")
    print("=" * 60)
    
    # Test imports
    try:
        from src import (
            CLIPEmbeddingExtractor,
            MedicalKnowledgeGraph,
            GraphRAGRetriever, 
            SSMQueryProcessor,
            get_available_models,
            check_huggingface_auth
        )
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test basic functionality
    try:
        print("\nTesting basic components...")
        
        # Test SSM
        ssm = SSMQueryProcessor()
        print("✅ SSM Processor initialized")
        
        # Test CLIP
        clip = CLIPEmbeddingExtractor()
        print("✅ CLIP Extractor initialized")
        
        # Test Graph
        graph = MedicalKnowledgeGraph()
        print("✅ Medical Knowledge Graph initialized")
        
        # Test Retriever
        retriever = GraphRAGRetriever(clip, ssm, graph)
        print("✅ GraphRAG Retriever initialized")
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n🎉 All tests passed! System is ready.")
    else:
        print("\n💥 Some tests failed. Check the errors above.")
        sys.exit(1)
