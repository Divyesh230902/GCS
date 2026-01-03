"""
State Space Model (SSM) for Query Handling and Data Retrieval
Uses Hugging Face transformers for state-space models
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    AutoConfig,
    MambaConfig,
    MambaForCausalLM
)
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class QueryResult:
    """Result from SSM query processing"""
    query: str
    processed_query: str
    intent: str
    confidence: float
    retrieval_instructions: Dict[str, Any]
    metadata: Dict[str, Any]

class SSMQueryProcessor:
    """
    State Space Model for processing queries and generating retrieval instructions
    """
    
    def __init__(self, 
                 model_key: str = "mamba-1.4b",
                 device: str = "auto",
                 max_length: int = 512):
        """
        Initialize SSM query processor
        
        Args:
            model_key: Key for model configuration (mamba-1.4b, mamba-130m, dialo-gpt-small, gpt2, rule-based)
            device: Device to run on ('auto', 'cpu', 'cuda')
            max_length: Maximum sequence length
        """
        from .model_config import get_model_config, check_huggingface_auth
        
        self.model_key = model_key
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        # Get model configuration
        self.model_config = get_model_config(model_key)
        self.model_name = self.model_config["model_name"]
        
        # Load tokenizer and model
        self._load_model()
        
        # Medical query patterns and intents
        self.intent_patterns = {
            "classification": [
                "classify", "identify", "diagnose", "detect", "recognize",
                "what disease", "what condition", "what type", "category"
            ],
            "retrieval": [
                "find", "search", "retrieve", "get", "show me", "look for",
                "similar", "like this", "examples of"
            ],
            "comparison": [
                "compare", "difference", "vs", "versus", "contrast",
                "better", "worse", "similarities"
            ],
            "analysis": [
                "analyze", "examine", "study", "investigate", "explore",
                "pattern", "trend", "characteristics"
            ]
        }
        
        # Medical domain keywords
        self.medical_keywords = [
            "alzheimer", "dementia", "brain tumor", "glioma", "meningioma",
            "parkinson", "parkinson's", "multiple sclerosis", "ms", "lesion",
            "mri", "scan", "imaging", "neurological", "cognitive"
        ]
    
    def _load_model(self):
        """Load the state-space model and tokenizer"""
        from .model_config import check_huggingface_auth
        
        # Check if we should use rule-based fallback
        if self.model_key == "rule-based" or self.model_name is None:
            print("Using rule-based query processing (no neural model)")
            self.model = None
            self.tokenizer = None
            return
        
        # Check authentication for gated models
        if self.model_config["requires_auth"] and not check_huggingface_auth():
            print(f"Model {self.model_name} requires Hugging Face authentication")
            print("Falling back to rule-based processing...")
            self.model = None
            self.tokenizer = None
            return
        
        try:
            print(f"Loading Mamba SSM model: {self.model_name}")
            print(f"Model type: {self.model_config['model_type']}")
            
            # Load tokenizer and model using the correct Mamba approach
            if "mamba" in self.model_name.lower():
                # Use MambaForCausalLM for Mamba models as per documentation
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = MambaForCausalLM.from_pretrained(self.model_name)
            else:
                # Use standard AutoModelForCausalLM for other models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Mamba SSM model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading Mamba SSM model: {e}")
            print("Falling back to rule-based processing...")
            
            # Fallback to rule-based processing
            self.model = None
            self.tokenizer = None
            print("Using rule-based query processing (no neural model)")
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> QueryResult:
        """
        Process a user query and generate retrieval instructions
        
        Args:
            query: User's natural language query
            context: Optional context about available data/metadata
            
        Returns:
            QueryResult with processed query and retrieval instructions
        """
        # Clean and preprocess query
        processed_query = self._preprocess_query(query)
        
        # Detect intent
        intent, confidence = self._detect_intent(processed_query)
        
        # Generate retrieval instructions
        retrieval_instructions = self._generate_retrieval_instructions(
            processed_query, intent, context
        )
        
        # Extract metadata
        metadata = self._extract_metadata(processed_query, intent)
        
        return QueryResult(
            query=query,
            processed_query=processed_query,
            intent=intent,
            confidence=confidence,
            retrieval_instructions=retrieval_instructions,
            metadata=metadata
        )
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and clean the query"""
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = " ".join(query.split())
        
        # Expand medical abbreviations
        abbreviations = {
            "ad": "alzheimer's disease",
            "pd": "parkinson's disease", 
            "ms": "multiple sclerosis",
            "bt": "brain tumor",
            "mri": "magnetic resonance imaging"
        }
        
        for abbr, full in abbreviations.items():
            query = query.replace(abbr, full)
        
        return query
    
    def _detect_intent(self, query: str) -> Tuple[str, float]:
        """Detect the intent of the query"""
        query_words = set(query.split())
        
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query:
                    score += 1
            intent_scores[intent] = score
        
        # Check for medical domain relevance
        medical_score = sum(1 for keyword in self.medical_keywords if keyword in query)
        
        if not intent_scores or max(intent_scores.values()) == 0:
            intent = "general"
            confidence = 0.3
        else:
            intent = max(intent_scores, key=intent_scores.get)
            confidence = min(0.9, 0.5 + (max(intent_scores.values()) * 0.1) + (medical_score * 0.05))
        
        return intent, confidence
    
    def _generate_retrieval_instructions(self, 
                                       query: str, 
                                       intent: str, 
                                       context: Optional[Dict]) -> Dict[str, Any]:
        """Generate specific instructions for data retrieval"""
        
        instructions = {
            "intent": intent,
            "query_type": "medical_image_retrieval",
            "filters": {},
            "sorting": {},
            "limit": 10,
            "similarity_threshold": 0.7
        }
        
        # Intent-specific instructions
        if intent == "classification":
            instructions["filters"]["task"] = "classification"
            instructions["limit"] = 5  # Fewer results for classification
            instructions["similarity_threshold"] = 0.8
            
        elif intent == "retrieval":
            instructions["filters"]["task"] = "similarity_search"
            instructions["limit"] = 20  # More results for retrieval
            instructions["similarity_threshold"] = 0.6
            
        elif intent == "comparison":
            instructions["filters"]["task"] = "comparison"
            instructions["limit"] = 10
            instructions["similarity_threshold"] = 0.7
            
        elif intent == "analysis":
            instructions["filters"]["task"] = "analysis"
            instructions["limit"] = 15
            instructions["similarity_threshold"] = 0.65
        
        # Extract specific medical conditions
        conditions = []
        for keyword in self.medical_keywords:
            if keyword in query:
                conditions.append(keyword)
        
        if conditions:
            instructions["filters"]["conditions"] = conditions
        
        # Extract dataset preferences
        if "alzheimer" in query or "dementia" in query:
            instructions["filters"]["preferred_dataset"] = "alzheimer"
        elif "tumor" in query or "glioma" in query or "meningioma" in query:
            instructions["filters"]["preferred_dataset"] = "brain_tumor"
        elif "parkinson" in query:
            instructions["filters"]["preferred_dataset"] = "parkinson"
        elif "sclerosis" in query or "ms" in query:
            instructions["filters"]["preferred_dataset"] = "ms"
        
        return instructions
    
    def _extract_metadata(self, query: str, intent: str) -> Dict[str, Any]:
        """Extract metadata from the query"""
        metadata = {
            "query_length": len(query.split()),
            "intent": intent,
            "contains_medical_terms": any(keyword in query for keyword in self.medical_keywords),
            "timestamp": torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        # Extract specific medical terms found
        found_terms = [term for term in self.medical_keywords if term in query]
        metadata["medical_terms"] = found_terms
        
        return metadata
    
    def generate_embedding_query(self, query: str) -> torch.Tensor:
        """
        Generate embedding for the query using the Mamba SSM model
        
        Args:
            query: Input query string
            
        Returns:
            Query embedding tensor
        """
        if self.model is None or self.tokenizer is None:
            # Fallback to simple TF-IDF like embedding
            return self._simple_embedding(query)
        
        # Tokenize input
        inputs = self.tokenizer(
            query, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        ).to(self.model.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # For Mamba models, use the appropriate output
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Use the last hidden state as query embedding
                query_embedding = outputs.hidden_states[-1].mean(dim=1)  # Average over sequence length
            elif hasattr(outputs, 'last_hidden_state'):
                query_embedding = outputs.last_hidden_state.mean(dim=1)
            else:
                # For Mamba models, use logits as embedding (as per documentation)
                query_embedding = outputs.logits.mean(dim=1)
            
        return query_embedding
    
    def _simple_embedding(self, query: str) -> torch.Tensor:
        """Simple fallback embedding based on medical keywords"""
        query_lower = query.lower()
        embedding = torch.zeros(len(self.medical_keywords))
        
        for i, keyword in enumerate(self.medical_keywords):
            if keyword in query_lower:
                embedding[i] = 1.0
        
        # Add intent-based features
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    embedding = torch.cat([embedding, torch.tensor([1.0])])
                    break
        
        return embedding.unsqueeze(0)  # Add batch dimension
    
    def batch_process_queries(self, queries: List[str]) -> List[QueryResult]:
        """Process multiple queries in batch"""
        results = []
        for query in queries:
            result = self.process_query(query)
            results.append(result)
        return results
    
    def save_query_history(self, results: List[QueryResult], filepath: str):
        """Save query processing history to file"""
        history = []
        for result in results:
            history.append({
                "query": result.query,
                "processed_query": result.processed_query,
                "intent": result.intent,
                "confidence": result.confidence,
                "retrieval_instructions": result.retrieval_instructions,
                "metadata": {k: v for k, v in result.metadata.items() if k != "timestamp"}
            })
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_query_history(self, filepath: str) -> List[QueryResult]:
        """Load query processing history from file"""
        with open(filepath, 'r') as f:
            history = json.load(f)
        
        results = []
        for item in history:
            result = QueryResult(
                query=item["query"],
                processed_query=item["processed_query"],
                intent=item["intent"],
                confidence=item["confidence"],
                retrieval_instructions=item["retrieval_instructions"],
                metadata=item["metadata"]
            )
            results.append(result)
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Initialize SSM processor
    ssm_processor = SSMQueryProcessor()
    
    # Test queries
    test_queries = [
        "Find similar brain tumor images",
        "Classify this MRI scan for Alzheimer's disease",
        "Compare Parkinson's vs normal brain scans",
        "Analyze the patterns in MS lesions",
        "Show me examples of glioma tumors"
    ]
    
    print("Testing SSM Query Processor:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = ssm_processor.process_query(query)
        print(f"Intent: {result.intent} (confidence: {result.confidence:.2f})")
        print(f"Retrieval Instructions: {result.retrieval_instructions}")
        print("-" * 30)
