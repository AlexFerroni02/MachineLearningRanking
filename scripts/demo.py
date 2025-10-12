#!/usr/bin/env python3
"""
Interactive Demo Script for Medical Laboratory Test Ranking

This script demonstrates the trained ranking model by allowing users to
search for medical laboratory tests.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ranker import LambdaMARTRanker, DatasetBuilder
from feature_extractor import FeatureExtractor


def format_document(doc: dict, rank: int, score: float) -> str:
    """Format a document for display.
    
    Args:
        doc: Document dictionary
        rank: Rank position
        score: Relevance score
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\n{rank}. {doc['title']}")
    lines.append(f"   Score: {score:.4f} | LOINC: {doc['loinc_code']} | Type: {doc['test_type']}")
    lines.append(f"   {doc['content'][:200]}...")
    return '\n'.join(lines)


def search_and_rank(query: str, ranker: LambdaMARTRanker, 
                   feature_extractor: FeatureExtractor,
                   documents: list, top_k: int = 5) -> list:
    """Search and rank documents for a query.
    
    Args:
        query: Search query
        ranker: Trained ranking model
        feature_extractor: Feature extractor
        documents: List of all documents
        top_k: Number of results to return
        
    Returns:
        List of (doc, score) tuples
    """
    # Extract features for all documents
    doc_ids = [doc['doc_id'] for doc in documents]
    features_list = []
    
    for doc_id in doc_ids:
        features = feature_extractor.extract_features(query, doc_id)
        features_list.append(features)
    
    # Get predictions
    X = np.array(features_list)
    scores = ranker.predict(X)
    
    # Sort by score (descending)
    ranked_indices = np.argsort(scores)[::-1]
    
    # Get top-k results
    results = []
    for idx in ranked_indices[:top_k]:
        results.append((documents[idx], scores[idx]))
    
    return results


def main():
    """Main demo function."""
    print("=" * 70)
    print("Medical Laboratory Test Ranking System - Interactive Demo")
    print("=" * 70)
    
    # Load data
    print("\nLoading data and model...")
    queries_data, documents_data, training_data = DatasetBuilder.load_data()
    documents = documents_data['documents']
    
    # Build feature extractor
    feature_extractor = FeatureExtractor(documents)
    
    # Load or train model
    model_path = Path(__file__).parent.parent / "model" / "lambdamart_ranker.txt"
    
    ranker = LambdaMARTRanker()
    
    if model_path.exists():
        print(f"Loading pre-trained model from {model_path}...")
        ranker.load_model(str(model_path))
        print("✓ Model loaded successfully!")
    else:
        print("No pre-trained model found. Training new model...")
        print("(This may take a minute...)\n")
        
        # Build feature matrix and train
        features, relevances, query_ids = DatasetBuilder.build_feature_matrix(
            training_data, queries_data, feature_extractor
        )
        
        ranker.train(
            features, relevances, query_ids,
            num_boost_round=100,
            feature_names=feature_extractor.get_feature_names()
        )
        
        # Save model
        model_path.parent.mkdir(exist_ok=True)
        ranker.save_model(str(model_path))
        print(f"✓ Model trained and saved to {model_path}")
    
    print("\n" + "=" * 70)
    print("System Ready!")
    print("=" * 70)
    
    # Show example queries
    print("\nExample queries:")
    for i, query in enumerate(queries_data['queries'][:5], 1):
        print(f"  {i}. {query['text']}")
    
    print("\n" + "-" * 70)
    
    # Interactive loop
    while True:
        print("\nEnter a search query (or 'quit' to exit):")
        query = input("> ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Medical Laboratory Test Ranking System!")
            break
        
        if not query:
            continue
        
        # Search and rank
        print(f"\nSearching for: '{query}'")
        print("=" * 70)
        
        try:
            results = search_and_rank(query, ranker, feature_extractor, documents, top_k=5)
            
            if results:
                print(f"\nTop 5 Results:")
                for rank, (doc, score) in enumerate(results, 1):
                    print(format_document(doc, rank, score))
            else:
                print("\nNo results found.")
        
        except Exception as e:
            print(f"\nError during search: {e}")
            continue
        
        print("\n" + "-" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
