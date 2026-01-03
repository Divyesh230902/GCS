#!/usr/bin/env python3
"""
GCS: Graph-based Classification System for Medical Image Retrieval
Main entry point for the system
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import (
    CLIPEmbeddingExtractor,
    MedicalKnowledgeGraph, 
    GraphRAGRetriever,
    SSMQueryProcessor,
    get_available_models,
    check_huggingface_auth
)

def main():
    """Main function to demonstrate GCS capabilities"""
    print("=" * 60)
    print("GCS: Graph-based Classification System")
    print("=" * 60)
    
    # Check available models
    print("\nAvailable models:")
    available_models = get_available_models()
    for key, desc in available_models.items():
        print(f"  {key}: {desc}")
    
    # Check authentication
    has_auth = check_huggingface_auth()
    print(f"\nHugging Face Authentication: {'✅ Available' if has_auth else '❌ Not Available'}")
    
    if not has_auth:
        print("\nTo use neural models, set up Hugging Face authentication:")
        print("  huggingface-cli login")
        print("  or")
        print("  export HUGGINGFACE_HUB_TOKEN=your_token_here")
    
    print("\n" + "=" * 60)
    print("System ready! Run tests to see the complete pipeline.")
    print("=" * 60)

if __name__ == "__main__":
    main()
