#!/usr/bin/env python3
"""
Training Script for LambdaMART Ranking Model

This script trains a Learning to Rank model using LambdaMART algorithm
on the medical laboratory test dataset.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ranker import LambdaMARTRanker, DatasetBuilder
from feature_extractor import FeatureExtractor
from evaluation import evaluate_model_on_queries


def main():
    """Main training function."""
    print("=" * 70)
    print("LambdaMART Training for Medical Laboratory Test Ranking")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading data...")
    queries_data, documents_data, training_data = DatasetBuilder.load_data()
    documents = documents_data['documents']
    
    print(f"  ✓ Loaded {len(queries_data['queries'])} queries")
    print(f"  ✓ Loaded {len(documents)} documents")
    print(f"  ✓ Loaded {len(training_data)} training examples")
    
    # Build feature extractor
    print("\n[2/5] Building feature extractor...")
    feature_extractor = FeatureExtractor(documents)
    feature_names = feature_extractor.get_feature_names()
    print(f"  ✓ Extracted {len(feature_names)} features per query-document pair")
    print(f"  Features: {', '.join(feature_names[:5])}...")
    
    # Build feature matrix
    print("\n[3/5] Building feature matrix...")
    features, relevances, query_ids = DatasetBuilder.build_feature_matrix(
        training_data, queries_data, feature_extractor
    )
    
    print(f"  ✓ Feature matrix shape: {features.shape}")
    print(f"  ✓ Relevance labels: {relevances.shape}")
    print(f"  ✓ Unique queries: {len(np.unique(query_ids))}")
    print(f"  ✓ Relevance distribution: {dict(zip(*np.unique(relevances, return_counts=True)))}")
    
    # Split into train/validation (80/20 by queries)
    unique_qids = np.unique(query_ids)
    np.random.seed(42)
    np.random.shuffle(unique_qids)
    
    split_idx = int(0.8 * len(unique_qids))
    train_qids = set(unique_qids[:split_idx])
    val_qids = set(unique_qids[split_idx:])
    
    train_mask = np.array([qid in train_qids for qid in query_ids])
    val_mask = np.array([qid in val_qids for qid in query_ids])
    
    train_features = features[train_mask]
    train_relevances = relevances[train_mask]
    train_query_ids = query_ids[train_mask]
    
    val_features = features[val_mask]
    val_relevances = relevances[val_mask]
    val_query_ids = query_ids[val_mask]
    
    print(f"\n  Split into train ({len(train_features)} examples) and validation ({len(val_features)} examples)")
    
    # Train model
    print("\n[4/5] Training LambdaMART model...")
    ranker = LambdaMARTRanker()
    
    history = ranker.train(
        train_features, train_relevances, train_query_ids,
        val_features, val_relevances, val_query_ids,
        num_boost_round=100,
        early_stopping_rounds=10,
        feature_names=feature_names
    )
    
    print("  ✓ Training complete!")
    
    # Display training metrics
    if 'train' in history:
        train_ndcg = history['train'].get('ndcg@10', [])
        if train_ndcg:
            print(f"  Final Train NDCG@10: {train_ndcg[-1]:.4f}")
    
    if 'valid' in history:
        val_ndcg = history['valid'].get('ndcg@10', [])
        if val_ndcg:
            print(f"  Final Valid NDCG@10: {val_ndcg[-1]:.4f}")
    
    # Feature importance
    print("\n[5/5] Analyzing feature importance...")
    importance = ranker.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n  Top 10 Important Features:")
    for i, (feat_name, imp) in enumerate(sorted_features[:10], 1):
        print(f"    {i:2d}. {feat_name:25s}: {imp:8.2f}")
    
    # Evaluate on all data
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    
    metrics = evaluate_model_on_queries(
        ranker, feature_extractor, queries_data, training_data
    )
    
    print("\nRanking Metrics (averaged across all queries):")
    print(f"  NDCG@1:  {metrics.get('ndcg@1', 0):.4f}")
    print(f"  NDCG@3:  {metrics.get('ndcg@3', 0):.4f}")
    print(f"  NDCG@5:  {metrics.get('ndcg@5', 0):.4f}")
    print(f"  NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
    print()
    print(f"  P@1:     {metrics.get('p@1', 0):.4f}")
    print(f"  P@3:     {metrics.get('p@3', 0):.4f}")
    print(f"  P@5:     {metrics.get('p@5', 0):.4f}")
    print(f"  P@10:    {metrics.get('p@10', 0):.4f}")
    print()
    print(f"  MAP:     {metrics.get('map', 0):.4f}")
    print(f"  MRR:     {metrics.get('mrr', 0):.4f}")
    
    # Save model
    model_path = Path(__file__).parent.parent / "model" / "lambdamart_ranker.txt"
    model_path.parent.mkdir(exist_ok=True)
    ranker.save_model(str(model_path))
    print(f"\n✓ Model saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return ranker, feature_extractor


if __name__ == '__main__':
    ranker, feature_extractor = main()
