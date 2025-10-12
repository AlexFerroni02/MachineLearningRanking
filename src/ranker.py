"""
Ranker Module

Implements Learning to Rank using LambdaMART algorithm via LightGBM.
"""

import numpy as np
import lightgbm as lgb
from typing import List, Tuple, Dict
import json
from pathlib import Path


class LambdaMARTRanker:
    """LambdaMART ranking model using LightGBM."""
    
    def __init__(self, params: Dict = None):
        """Initialize LambdaMART ranker.
        
        Args:
            params: LightGBM parameters (optional)
        """
        if params is None:
            # Default parameters optimized for ranking
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [1, 3, 5, 10],
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 6,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1
            }
        
        self.params = params
        self.model = None
        self.feature_names = None
    
    def prepare_training_data(self, features: np.ndarray, 
                             relevances: np.ndarray,
                             query_ids: np.ndarray) -> lgb.Dataset:
        """Prepare data for LightGBM training.
        
        Args:
            features: Feature matrix (n_samples x n_features)
            relevances: Relevance labels (n_samples,)
            query_ids: Query IDs for grouping (n_samples,)
            
        Returns:
            LightGBM Dataset
        """
        # Count documents per query for group information
        unique_qids, group_counts = np.unique(query_ids, return_counts=True)
        
        # Create dataset
        train_data = lgb.Dataset(
            features,
            label=relevances,
            group=group_counts,
            feature_name=self.feature_names
        )
        
        return train_data
    
    def train(self, train_features: np.ndarray, train_relevances: np.ndarray,
              train_query_ids: np.ndarray, val_features: np.ndarray = None,
              val_relevances: np.ndarray = None, val_query_ids: np.ndarray = None,
              num_boost_round: int = 100, early_stopping_rounds: int = 10,
              feature_names: List[str] = None) -> Dict:
        """Train the LambdaMART model.
        
        Args:
            train_features: Training feature matrix
            train_relevances: Training relevance labels
            train_query_ids: Training query IDs
            val_features: Validation feature matrix (optional)
            val_relevances: Validation relevance labels (optional)
            val_query_ids: Validation query IDs (optional)
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
            feature_names: Names of features
            
        Returns:
            Training history dictionary
        """
        self.feature_names = feature_names
        
        # Prepare training data
        train_data = self.prepare_training_data(
            train_features, train_relevances, train_query_ids
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        # Prepare validation data if provided
        if val_features is not None and val_relevances is not None:
            val_data = self.prepare_training_data(
                val_features, val_relevances, val_query_ids
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        evals_result = {}
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        return evals_result
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict relevance scores for documents.
        
        Args:
            features: Feature matrix (n_samples x n_features)
            
        Returns:
            Predicted scores (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(features)
    
    def rank_documents(self, query_features: np.ndarray, 
                      doc_ids: List[str] = None) -> List[Tuple[int, float]]:
        """Rank documents for a query.
        
        Args:
            query_features: Feature matrix for query-document pairs
            doc_ids: Optional document IDs
            
        Returns:
            List of (index, score) tuples sorted by score (descending)
        """
        scores = self.predict(query_features)
        
        # Sort by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        
        if doc_ids:
            return [(doc_ids[i], scores[i]) for i in ranked_indices]
        else:
            return [(i, scores[i]) for i in ranked_indices]
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('split', 'gain')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}
    
    def save_model(self, filepath: str):
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load model from file.
        
        Args:
            filepath: Path to model file
        """
        self.model = lgb.Booster(model_file=filepath)


class DatasetBuilder:
    """Builds training/validation datasets for ranking."""
    
    @staticmethod
    def load_data(data_dir: str = None) -> Tuple[Dict, Dict, List[Dict]]:
        """Load queries, documents, and training data.
        
        Args:
            data_dir: Directory containing data files
            
        Returns:
            Tuple of (queries_dict, documents_dict, training_data_list)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        else:
            data_dir = Path(data_dir)
        
        # Load queries
        with open(data_dir / "queries.json", 'r') as f:
            queries_data = json.load(f)
        
        # Load documents
        with open(data_dir / "documents.json", 'r') as f:
            documents_data = json.load(f)
        
        # Load training data
        with open(data_dir / "training_data.json", 'r') as f:
            training_data_raw = json.load(f)
            training_data = training_data_raw['training_data']
        
        return queries_data, documents_data, training_data
    
    @staticmethod
    def build_feature_matrix(training_data: List[Dict], queries_data: Dict,
                            feature_extractor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build feature matrix from training data.
        
        Args:
            training_data: List of training examples
            queries_data: Dictionary with queries
            feature_extractor: FeatureExtractor instance
            
        Returns:
            Tuple of (features, relevances, query_ids)
        """
        # Create query lookup
        query_lookup = {q['qid']: q['text'] for q in queries_data['queries']}
        
        # Map query IDs to integers
        qid_to_int = {qid: i for i, qid in enumerate(sorted(set(query_lookup.keys())))}
        
        features_list = []
        relevances_list = []
        query_ids_list = []
        
        for item in training_data:
            qid = item['qid']
            doc_id = item['doc_id']
            relevance = item['relevance']
            
            if qid not in query_lookup:
                continue
            
            query_text = query_lookup[qid]
            
            try:
                # Extract features
                features = feature_extractor.extract_features(query_text, doc_id)
                
                features_list.append(features)
                relevances_list.append(relevance)
                query_ids_list.append(qid_to_int[qid])
            except ValueError:
                # Document not found, skip
                continue
        
        return (
            np.array(features_list),
            np.array(relevances_list),
            np.array(query_ids_list)
        )


if __name__ == '__main__':
    # Example usage
    from feature_extractor import FeatureExtractor
    
    print("LambdaMART Ranker Demo")
    print("=" * 50)
    
    # Load data
    queries_data, documents_data, training_data = DatasetBuilder.load_data()
    documents = documents_data['documents']
    
    print(f"\nLoaded {len(queries_data['queries'])} queries")
    print(f"Loaded {len(documents)} documents")
    print(f"Loaded {len(training_data)} training examples")
    
    # Build feature extractor
    feature_extractor = FeatureExtractor(documents)
    
    # Build feature matrix
    print("\nBuilding feature matrix...")
    features, relevances, query_ids = DatasetBuilder.build_feature_matrix(
        training_data, queries_data, feature_extractor
    )
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Relevances shape: {relevances.shape}")
    print(f"Unique queries: {len(np.unique(query_ids))}")
    
    # Train model
    print("\nTraining LambdaMART model...")
    ranker = LambdaMARTRanker()
    history = ranker.train(
        features, relevances, query_ids,
        num_boost_round=50,
        feature_names=feature_extractor.get_feature_names()
    )
    
    print("Training complete!")
    
    # Show feature importance
    print("\nTop 5 Important Features:")
    importance = ranker.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat_name, imp in sorted_features[:5]:
        print(f"  {feat_name:25s}: {imp:.2f}")
