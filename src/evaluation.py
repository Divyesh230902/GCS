"""
Evaluation Framework for GraphRAG System
Calculates real metrics: Precision@K, Recall@K, NDCG@K, MAP
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time

@dataclass
class RetrievalMetrics:
    """Storage for retrieval metrics"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    map_score: float
    mrr: float  # Mean Reciprocal Rank
    query_time: float  # milliseconds
    
    def to_dict(self):
        return {
            'precision@k': self.precision_at_k,
            'recall@k': self.recall_at_k,
            'ndcg@k': self.ndcg_at_k,
            'map': self.map_score,
            'mrr': self.mrr,
            'query_time_ms': self.query_time
        }

class GraphRAGEvaluator:
    """
    Comprehensive evaluation framework for GraphRAG system
    """
    
    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluator
        
        Args:
            k_values: List of K values to evaluate (e.g., [1, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.results = {}
        
    def evaluate_retrieval(self,
                          retrieved_items: List[str],
                          relevant_items: Set[str],
                          k_values: List[int] = None) -> RetrievalMetrics:
        """
        Evaluate a single retrieval result
        
        Args:
            retrieved_items: List of retrieved item IDs (ordered by rank)
            relevant_items: Set of ground truth relevant item IDs
            k_values: K values to evaluate
            
        Returns:
            RetrievalMetrics object
        """
        k_values = k_values or self.k_values
        
        # Calculate metrics at different K values
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in k_values:
            # Precision@K
            retrieved_at_k = retrieved_items[:k]
            relevant_retrieved = len([item for item in retrieved_at_k if item in relevant_items])
            precision_at_k[k] = relevant_retrieved / k if k > 0 else 0.0
            
            # Recall@K
            recall_at_k[k] = relevant_retrieved / len(relevant_items) if len(relevant_items) > 0 else 0.0
            
            # NDCG@K
            ndcg_at_k[k] = self._calculate_ndcg(retrieved_items[:k], relevant_items)
        
        # MAP (Mean Average Precision)
        map_score = self._calculate_map(retrieved_items, relevant_items)
        
        # MRR (Mean Reciprocal Rank)
        mrr = self._calculate_mrr(retrieved_items, relevant_items)
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            map_score=map_score,
            mrr=mrr,
            query_time=0.0  # Set externally
        )
    
    def _calculate_ndcg(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not relevant:
            return 0.0
        
        # DCG: sum of relevance / log2(position + 1)
        dcg = 0.0
        for i, item in enumerate(retrieved):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # +2 because positions start at 1
        
        # IDCG: ideal DCG (all relevant items at top)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(retrieved), len(relevant))))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_map(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Mean Average Precision"""
        if not relevant:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, item in enumerate(retrieved):
            if item in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant) if len(relevant) > 0 else 0.0
    
    def _calculate_mrr(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, item in enumerate(retrieved):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_queries(self,
                        retriever,
                        queries: List[Dict],
                        ground_truth: Dict[str, Set[str]]) -> Dict[str, RetrievalMetrics]:
        """
        Evaluate multiple queries
        
        Args:
            retriever: Retrieval system (Enhanced GraphRAG or baseline)
            queries: List of query dicts with 'id', 'text', 'mode' (optional)
            ground_truth: Dict mapping query_id to set of relevant item IDs
            
        Returns:
            Dict mapping query_id to RetrievalMetrics
        """
        results = {}
        
        for query in queries:
            query_id = query['id']
            query_text = query['text']
            search_mode = query.get('mode', 'auto')
            
            print(f"Evaluating query {query_id}: {query_text[:50]}...")
            
            # Perform retrieval and time it
            start_time = time.time()
            result = retriever.retrieve(query_text, top_k=max(self.k_values), search_mode=search_mode)
            query_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Extract retrieved item IDs
            retrieved_items = [img.image_path for img in result.retrieved_images]
            
            # Get ground truth
            relevant_items = ground_truth.get(query_id, set())
            
            # Calculate metrics
            metrics = self.evaluate_retrieval(retrieved_items, relevant_items)
            metrics.query_time = query_time
            
            results[query_id] = metrics
        
        return results
    
    def aggregate_metrics(self, query_results: Dict[str, RetrievalMetrics]) -> Dict:
        """
        Aggregate metrics across all queries
        
        Args:
            query_results: Dict mapping query_id to RetrievalMetrics
            
        Returns:
            Dict with averaged metrics
        """
        if not query_results:
            return {}
        
        aggregated = {
            'precision@k': defaultdict(list),
            'recall@k': defaultdict(list),
            'ndcg@k': defaultdict(list),
            'map': [],
            'mrr': [],
            'query_time_ms': []
        }
        
        # Collect all values
        for metrics in query_results.values():
            for k, val in metrics.precision_at_k.items():
                aggregated['precision@k'][k].append(val)
            for k, val in metrics.recall_at_k.items():
                aggregated['recall@k'][k].append(val)
            for k, val in metrics.ndcg_at_k.items():
                aggregated['ndcg@k'][k].append(val)
            aggregated['map'].append(metrics.map_score)
            aggregated['mrr'].append(metrics.mrr)
            aggregated['query_time_ms'].append(metrics.query_time)
        
        # Calculate averages
        result = {
            'precision@k': {k: np.mean(vals) for k, vals in aggregated['precision@k'].items()},
            'recall@k': {k: np.mean(vals) for k, vals in aggregated['recall@k'].items()},
            'ndcg@k': {k: np.mean(vals) for k, vals in aggregated['ndcg@k'].items()},
            'map': np.mean(aggregated['map']),
            'mrr': np.mean(aggregated['mrr']),
            'avg_query_time_ms': np.mean(aggregated['query_time_ms']),
            'throughput_qps': 1000.0 / np.mean(aggregated['query_time_ms']) if aggregated['query_time_ms'] else 0
        }
        
        return result
    
    def format_for_plots(self, aggregated_metrics: Dict) -> Dict:
        """
        Format aggregated metrics for visualization
        
        Args:
            aggregated_metrics: Output from aggregate_metrics
            
        Returns:
            Dict formatted for plot functions
        """
        # Convert to lists for plotting
        k_values = sorted(aggregated_metrics['precision@k'].keys())
        
        return {
            'precision@k': [aggregated_metrics['precision@k'][k] for k in k_values],
            'recall@k': [aggregated_metrics['recall@k'][k] for k in k_values],
            'ndcg@k': [aggregated_metrics['ndcg@k'][k] for k in k_values],
            'map': aggregated_metrics['map']
        }


class AblationStudyEvaluator:
    """
    Specialized evaluator for ablation studies
    """
    
    def __init__(self, base_evaluator: GraphRAGEvaluator):
        """
        Initialize ablation study evaluator
        
        Args:
            base_evaluator: Base GraphRAGEvaluator instance
        """
        self.base_evaluator = base_evaluator
        self.ablation_results = {}
    
    def evaluate_component_ablation(self,
                                    retrievers: Dict[str, any],
                                    queries: List[Dict],
                                    ground_truth: Dict[str, Set[str]]) -> Dict:
        """
        Evaluate component ablation study
        
        Args:
            retrievers: Dict of {config_name: retriever_instance}
            queries: List of queries
            ground_truth: Ground truth relevance
            
        Returns:
            Dict formatted for ablation plots
        """
        print("\n" + "=" * 70)
        print("COMPONENT ABLATION STUDY")
        print("=" * 70)
        
        results = {}
        
        for config_name, retriever in retrievers.items():
            print(f"\nEvaluating: {config_name}")
            
            # Evaluate all queries
            query_results = self.base_evaluator.evaluate_queries(retriever, queries, ground_truth)
            
            # Aggregate
            aggregated = self.base_evaluator.aggregate_metrics(query_results)
            
            # Store averaged metrics at K=5 for simplicity
            results[config_name] = {
                'precision': aggregated['precision@k'].get(5, 0.0),
                'recall': aggregated['recall@k'].get(5, 0.0),
                'ndcg': aggregated['ndcg@k'].get(5, 0.0),
                'f1': 2 * aggregated['precision@k'].get(5, 0.0) * aggregated['recall@k'].get(5, 0.0) / 
                      (aggregated['precision@k'].get(5, 0.0) + aggregated['recall@k'].get(5, 0.0) + 1e-10)
            }
            
            print(f"  Precision@5: {results[config_name]['precision']:.3f}")
            print(f"  Recall@5: {results[config_name]['recall']:.3f}")
            print(f"  NDCG@5: {results[config_name]['ndcg']:.3f}")
        
        self.ablation_results['component'] = results
        return results
    
    def evaluate_hierarchy_ablation(self,
                                    retrievers: Dict[int, any],
                                    queries: List[Dict],
                                    ground_truth: Dict[str, Set[str]]) -> Dict:
        """
        Evaluate hierarchy level ablation
        
        Args:
            retrievers: Dict of {num_levels: retriever_instance}
            queries: List of queries
            ground_truth: Ground truth relevance
            
        Returns:
            Dict formatted for hierarchy ablation plots
        """
        print("\n" + "=" * 70)
        print("HIERARCHY LEVEL ABLATION STUDY")
        print("=" * 70)
        
        results = {}
        
        for num_levels, retriever in sorted(retrievers.items()):
            print(f"\nEvaluating: {num_levels} level(s)")
            
            query_results = self.base_evaluator.evaluate_queries(retriever, queries, ground_truth)
            aggregated = self.base_evaluator.aggregate_metrics(query_results)
            
            results[num_levels] = {
                'precision': aggregated['precision@k'].get(5, 0.0),
                'recall': aggregated['recall@k'].get(5, 0.0),
                'ndcg': aggregated['ndcg@k'].get(5, 0.0),
                'f1': 2 * aggregated['precision@k'].get(5, 0.0) * aggregated['recall@k'].get(5, 0.0) / 
                      (aggregated['precision@k'].get(5, 0.0) + aggregated['recall@k'].get(5, 0.0) + 1e-10)
            }
            
            print(f"  Precision@5: {results[num_levels]['precision']:.3f}")
            print(f"  Recall@5: {results[num_levels]['recall']:.3f}")
        
        self.ablation_results['hierarchy'] = results
        return results
    
    def evaluate_search_mode_ablation(self,
                                     retriever,
                                     queries_by_mode: Dict[str, List[Dict]],
                                     ground_truth: Dict[str, Set[str]]) -> Dict:
        """
        Evaluate search mode ablation (Global/Local/Hybrid/Auto)
        
        Args:
            retriever: Single retriever supporting multiple modes
            queries_by_mode: Dict of {mode: queries_for_that_mode}
            ground_truth: Ground truth relevance
            
        Returns:
            Dict formatted for search mode plots
        """
        print("\n" + "=" * 70)
        print("SEARCH MODE ABLATION STUDY")
        print("=" * 70)
        
        results = {}
        
        for mode, queries in queries_by_mode.items():
            print(f"\nEvaluating: {mode} search mode")
            
            # Set mode for all queries
            for query in queries:
                query['mode'] = mode.lower()
            
            query_results = self.base_evaluator.evaluate_queries(retriever, queries, ground_truth)
            aggregated = self.base_evaluator.aggregate_metrics(query_results)
            
            results[mode] = {
                'precision': aggregated['precision@k'].get(5, 0.0),
                'recall': aggregated['recall@k'].get(5, 0.0),
                'ndcg': aggregated['ndcg@k'].get(5, 0.0),
                'f1': 2 * aggregated['precision@k'].get(5, 0.0) * aggregated['recall@k'].get(5, 0.0) / 
                      (aggregated['precision@k'].get(5, 0.0) + aggregated['recall@k'].get(5, 0.0) + 1e-10)
            }
            
            print(f"  Precision@5: {results[mode]['precision']:.3f}")
            print(f"  Recall@5: {results[mode]['recall']:.3f}")
        
        self.ablation_results['search_mode'] = results
        return results
    
    def save_results(self, output_path: str):
        """Save ablation results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.ablation_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")


class GroundTruthGenerator:
    """
    Helper to generate ground truth for evaluation
    """
    
    @staticmethod
    def generate_from_class_labels(image_embeddings: List,
                                   queries: List[Dict]) -> Dict[str, Set[str]]:
        """
        Generate ground truth based on class label matching
        
        Args:
            image_embeddings: List of ImageEmbedding objects
            queries: List of query dicts with 'id' and 'target_class'
            
        Returns:
            Dict mapping query_id to set of relevant image paths
        """
        ground_truth = {}
        
        # Group images by class
        images_by_class = defaultdict(list)
        for img_emb in image_embeddings:
            images_by_class[img_emb.class_label].append(img_emb.image_path)
        
        # For each query, find relevant images
        for query in queries:
            query_id = query['id']
            target_class = query.get('target_class')
            target_dataset = query.get('target_dataset')
            
            if target_class:
                # Images with matching class are relevant
                relevant = set(images_by_class.get(target_class, []))
                
                # If dataset specified, further filter
                if target_dataset:
                    relevant = {img for img in relevant 
                               if any(emb.image_path == img and emb.dataset == target_dataset 
                                     for emb in image_embeddings)}
                
                ground_truth[query_id] = relevant
            else:
                # If no specific class, use dataset
                if target_dataset:
                    relevant = {emb.image_path for emb in image_embeddings 
                               if emb.dataset == target_dataset}
                    ground_truth[query_id] = relevant
        
        return ground_truth
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Set[str]]:
        """Load ground truth from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists to sets
        return {qid: set(items) for qid, items in data.items()}
    
    @staticmethod
    def save_to_file(ground_truth: Dict[str, Set[str]], filepath: str):
        """Save ground truth to JSON file"""
        # Convert sets to lists for JSON
        data = {qid: list(items) for qid, items in ground_truth.items()}
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    print("GraphRAG Evaluation Framework")
    print("Calculates Precision@K, Recall@K, NDCG@K, MAP for real experiments")

