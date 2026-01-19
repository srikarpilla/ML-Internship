import numpy as np
from typing import List, Dict, Tuple

class RetrievalMetrics:
    """Evaluation metrics for image retrieval"""
    
    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: List[int], k: int = 10) -> float:
        """
        Precision@K: Proportion of retrieved items that are relevant
        
        Args:
            retrieved: List of retrieved item IDs
            relevant: List of relevant item IDs
            k: Cutoff position
        """
        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)
        
        if not retrieved_at_k:
            return 0.0
        
        num_relevant = len([item for item in retrieved_at_k if item in relevant_set])
        return num_relevant / len(retrieved_at_k)
    
    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: List[int], k: int = 10) -> float:
        """
        Recall@K: Proportion of relevant items that are retrieved
        """
        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)
        
        if not relevant:
            return 0.0
        
        num_relevant = len([item for item in retrieved_at_k if item in relevant_set])
        return num_relevant / len(relevant)
    
    @staticmethod
    def average_precision(retrieved: List[int], relevant: List[int]) -> float:
        """
        Average Precision: Area under precision-recall curve
        """
        relevant_set = set(relevant)
        
        if not relevant:
            return 0.0
        
        score = 0.0
        num_relevant = 0
        
        for i, item in enumerate(retrieved, 1):
            if item in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / i
                score += precision_at_i
        
        return score / len(relevant) if relevant else 0.0
    
    @staticmethod
    def mean_average_precision(results: Dict[str, List[int]], 
                               ground_truth: Dict[str, List[int]]) -> float:
        """
        Mean Average Precision across multiple queries
        
        Args:
            results: Dict mapping query to list of retrieved IDs
            ground_truth: Dict mapping query to list of relevant IDs
        """
        aps = []
        
        for query in results:
            if query in ground_truth:
                ap = RetrievalMetrics.average_precision(
                    results[query], 
                    ground_truth[query]
                )
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[int], relevant: List[int], 
                  k: int = 10, relevance_scores: Dict[int, float] = None) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        Assumes binary relevance if relevance_scores not provided
        """
        retrieved_at_k = retrieved[:k]
        
        if not retrieved_at_k:
            return 0.0
        
        # Use binary relevance if scores not provided
        if relevance_scores is None:
            relevant_set = set(relevant)
            relevance_scores = {item: 1.0 for item in relevant_set}
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(retrieved_at_k, 1):
            rel = relevance_scores.get(item, 0.0)
            dcg += rel / np.log2(i + 1)
        
        # Calculate IDCG
        ideal_scores = sorted(
            [relevance_scores.get(item, 0.0) for item in relevant[:k]], 
            reverse=True
        )
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(results: Dict[str, List[int]], 
                            ground_truth: Dict[str, List[int]]) -> float:
        """
        Mean Reciprocal Rank: Average of reciprocal ranks of first relevant item
        """
        rr_scores = []
        
        for query in results:
            if query not in ground_truth:
                continue
            
            relevant_set = set(ground_truth[query])
            retrieved = results[query]
            
            for i, item in enumerate(retrieved, 1):
                if item in relevant_set:
                    rr_scores.append(1.0 / i)
                    break
            else:
                rr_scores.append(0.0)
        
        return np.mean(rr_scores) if rr_scores else 0.0
    
    @staticmethod
    def evaluate_retrieval(retrieved: List[int], relevant: List[int], 
                          k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Comprehensive evaluation with multiple metrics
        
        Returns:
            Dictionary with precision@k, recall@k, AP, and nDCG@k
        """
        metrics = {
            'average_precision': RetrievalMetrics.average_precision(retrieved, relevant)
        }
        
        for k in k_values:
            metrics[f'precision@{k}'] = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
            metrics[f'recall@{k}'] = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
            metrics[f'ndcg@{k}'] = RetrievalMetrics.ndcg_at_k(retrieved, relevant, k)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Retrieval Metrics"):
        """Pretty print metrics"""
        print(f"\n{title}")
        print("=" * 60)
        
        for metric_name, value in metrics.items():
            print(f"{metric_name:.<40} {value:.4f}")
        
        print("=" * 60)