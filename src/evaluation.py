"""
Evaluation Metrics Module

Implements standard Information Retrieval metrics for ranking evaluation.
"""

import numpy as np
from typing import List, Dict


class RankingMetrics:
    """Computes ranking evaluation metrics."""
    
    @staticmethod
    def dcg(relevances: List[float], k: int = None) -> float:
        """Compute Discounted Cumulative Gain.
        
        Args:
            relevances: List of relevance scores in ranked order
            k: Cut-off position (None for all)
            
        Returns:
            DCG score
        """
        if k is not None:
            relevances = relevances[:k]
        
        dcg_score = 0.0
        for i, rel in enumerate(relevances):
            # DCG formula: sum(rel_i / log2(i+2))
            dcg_score += rel / np.log2(i + 2)
        
        return dcg_score
    
    @staticmethod
    def ndcg(relevances: List[float], k: int = None) -> float:
        """Compute Normalized Discounted Cumulative Gain.
        
        Args:
            relevances: List of relevance scores in ranked order
            k: Cut-off position (None for all)
            
        Returns:
            NDCG score (0 to 1)
        """
        if not relevances:
            return 0.0
        
        # Compute DCG
        dcg_score = RankingMetrics.dcg(relevances, k)
        
        # Compute ideal DCG (sort relevances in descending order)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg_score = RankingMetrics.dcg(ideal_relevances, k)
        
        # Normalize
        if idcg_score == 0:
            return 0.0
        
        return dcg_score / idcg_score
    
    @staticmethod
    def precision_at_k(relevances: List[float], k: int, threshold: float = 1.0) -> float:
        """Compute Precision at K.
        
        Args:
            relevances: List of relevance scores in ranked order
            k: Cut-off position
            threshold: Minimum relevance to be considered relevant
            
        Returns:
            Precision@K score
        """
        if k <= 0 or not relevances:
            return 0.0
        
        top_k = relevances[:k]
        relevant_count = sum(1 for rel in top_k if rel >= threshold)
        return relevant_count / k
    
    @staticmethod
    def average_precision(relevances: List[float], threshold: float = 1.0) -> float:
        """Compute Average Precision.
        
        Args:
            relevances: List of relevance scores in ranked order
            threshold: Minimum relevance to be considered relevant
            
        Returns:
            AP score
        """
        if not relevances:
            return 0.0
        
        num_relevant = sum(1 for rel in relevances if rel >= threshold)
        if num_relevant == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, rel in enumerate(relevances):
            if rel >= threshold:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / num_relevant
    
    @staticmethod
    def reciprocal_rank(relevances: List[float], threshold: float = 1.0) -> float:
        """Compute Reciprocal Rank.
        
        Args:
            relevances: List of relevance scores in ranked order
            threshold: Minimum relevance to be considered relevant
            
        Returns:
            RR score
        """
        for i, rel in enumerate(relevances):
            if rel >= threshold:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def evaluate_ranking(true_relevances: List[float], 
                        predicted_scores: List[float],
                        k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Evaluate a ranking with multiple metrics.
        
        Args:
            true_relevances: True relevance scores
            predicted_scores: Predicted scores from model
            k_values: List of k values for NDCG@k and P@k
            
        Returns:
            Dictionary of metric names to scores
        """
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevances = [true_relevances[i] for i in sorted_indices]
        
        metrics = {}
        
        # NDCG at various k values
        for k in k_values:
            metrics[f'ndcg@{k}'] = RankingMetrics.ndcg(sorted_relevances, k)
        
        # Precision at various k values
        for k in k_values:
            metrics[f'p@{k}'] = RankingMetrics.precision_at_k(sorted_relevances, k)
        
        # Overall metrics
        metrics['map'] = RankingMetrics.average_precision(sorted_relevances)
        metrics['mrr'] = RankingMetrics.reciprocal_rank(sorted_relevances)
        
        return metrics


def evaluate_model_on_queries(model, feature_extractor, queries_data: Dict, 
                               training_data: List[Dict]) -> Dict[str, float]:
    """Evaluate model performance across all queries.
    
    Args:
        model: Trained ranking model
        feature_extractor: FeatureExtractor instance
        queries_data: Dictionary with queries
        training_data: List of query-document-relevance tuples
        
    Returns:
        Dictionary of averaged metrics
    """
    from collections import defaultdict
    
    # Group training data by query
    query_docs = defaultdict(list)
    for item in training_data:
        query_docs[item['qid']].append({
            'doc_id': item['doc_id'],
            'relevance': item['relevance']
        })
    
    # Create query lookup
    query_lookup = {q['qid']: q['text'] for q in queries_data['queries']}
    
    all_metrics = defaultdict(list)
    
    # Evaluate each query
    for qid, docs in query_docs.items():
        if qid not in query_lookup:
            continue
        
        query_text = query_lookup[qid]
        
        # Extract features and get predictions
        doc_ids = [d['doc_id'] for d in docs]
        true_relevances = [d['relevance'] for d in docs]
        
        features_list = []
        for doc_id in doc_ids:
            try:
                features = feature_extractor.extract_features(query_text, doc_id)
                features_list.append(features)
            except ValueError:
                # Document not found, skip
                continue
        
        if not features_list:
            continue
        
        # Predict scores
        X = np.array(features_list)
        predicted_scores = model.predict(X)
        
        # Compute metrics for this query
        metrics = RankingMetrics.evaluate_ranking(
            true_relevances, 
            predicted_scores,
            k_values=[1, 3, 5, 10]
        )
        
        # Accumulate metrics
        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)
    
    # Average metrics across queries
    avg_metrics = {
        metric_name: np.mean(values) 
        for metric_name, values in all_metrics.items()
    }
    
    return avg_metrics


if __name__ == '__main__':
    # Example usage
    print("Ranking Metrics Demo")
    print("=" * 50)
    
    # Example relevances (in ranked order by predicted scores)
    relevances = [3, 2, 0, 3, 1, 0, 2, 0]
    
    print(f"\nRelevance scores (ranked order): {relevances}")
    
    # Compute metrics
    print(f"\nNDCG@3: {RankingMetrics.ndcg(relevances, 3):.4f}")
    print(f"NDCG@5: {RankingMetrics.ndcg(relevances, 5):.4f}")
    print(f"NDCG@10: {RankingMetrics.ndcg(relevances, 10):.4f}")
    
    print(f"\nP@3: {RankingMetrics.precision_at_k(relevances, 3):.4f}")
    print(f"P@5: {RankingMetrics.precision_at_k(relevances, 5):.4f}")
    
    print(f"\nMAP: {RankingMetrics.average_precision(relevances):.4f}")
    print(f"MRR: {RankingMetrics.reciprocal_rank(relevances):.4f}")
    
    # Example with predicted scores
    print("\n" + "=" * 50)
    print("Evaluation with predicted scores")
    
    true_relevances = [3, 1, 0, 2, 0, 1]
    predicted_scores = [0.9, 0.4, 0.2, 0.8, 0.1, 0.5]
    
    metrics = RankingMetrics.evaluate_ranking(true_relevances, predicted_scores)
    
    print("\nMetrics:")
    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name}: {value:.4f}")
