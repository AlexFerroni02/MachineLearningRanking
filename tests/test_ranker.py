"""
Unit tests for the ranking system components.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loinc_mapper import LOINCMapper
from feature_extractor import FeatureExtractor
from evaluation import RankingMetrics
from ranker import DatasetBuilder


class TestLOINCMapper(unittest.TestCase):
    """Test LOINC mapping functionality."""
    
    def setUp(self):
        self.mapper = LOINCMapper()
    
    def test_get_loinc_code(self):
        """Test getting LOINC code."""
        code = self.mapper.get_loinc_code('glucose_in_blood')
        self.assertEqual(code, '2345-7')
    
    def test_search_by_keyword(self):
        """Test keyword search."""
        results = self.mapper.search_by_keyword('glucose')
        self.assertGreater(len(results), 0)
    
    def test_get_synonyms(self):
        """Test getting synonyms."""
        synonyms = self.mapper.get_synonyms('glucose_in_blood')
        self.assertIn('blood glucose', synonyms)


class TestFeatureExtractor(unittest.TestCase):
    """Test feature extraction."""
    
    def setUp(self):
        # Load documents
        _, documents_data, _ = DatasetBuilder.load_data()
        self.documents = documents_data['documents']
        self.extractor = FeatureExtractor(self.documents)
    
    def test_extract_features(self):
        """Test feature extraction."""
        query = "glucose in blood"
        doc_id = "D1"
        
        features = self.extractor.extract_features(query, doc_id)
        
        # Should return correct number of features
        self.assertEqual(len(features), len(self.extractor.get_feature_names()))
        
        # Features should be numeric
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_feature_values(self):
        """Test that feature values are reasonable."""
        query = "glucose in blood"
        doc_id = "D1"  # Highly relevant document
        
        features = self.extractor.extract_features(query, doc_id)
        
        # TF-IDF and BM25 should be positive for relevant doc
        self.assertGreater(features[0], 0)  # tfidf_content
        self.assertGreater(features[2], 0)  # bm25_content


class TestRankingMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def test_ndcg(self):
        """Test NDCG calculation."""
        relevances = [3, 2, 1, 0]
        ndcg = RankingMetrics.ndcg(relevances, k=None)
        
        # Perfect ranking should have NDCG = 1.0
        self.assertAlmostEqual(ndcg, 1.0, places=5)
    
    def test_ndcg_imperfect(self):
        """Test NDCG with imperfect ranking."""
        relevances = [0, 3, 2, 1]
        ndcg = RankingMetrics.ndcg(relevances, k=None)
        
        # Imperfect ranking should have NDCG < 1.0
        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0.0)
    
    def test_precision_at_k(self):
        """Test Precision@k."""
        relevances = [3, 2, 0, 0, 1]
        p_at_3 = RankingMetrics.precision_at_k(relevances, 3, threshold=1.0)
        
        # Top 3 has 2 relevant docs (relevance >= 1)
        self.assertAlmostEqual(p_at_3, 2/3, places=5)
    
    def test_mrr(self):
        """Test Mean Reciprocal Rank."""
        relevances = [0, 0, 3, 1]
        mrr = RankingMetrics.reciprocal_rank(relevances, threshold=1.0)
        
        # First relevant at position 3 (index 2)
        self.assertAlmostEqual(mrr, 1/3, places=5)


class TestDatasetBuilder(unittest.TestCase):
    """Test dataset building."""
    
    def test_load_data(self):
        """Test data loading."""
        queries_data, documents_data, training_data = DatasetBuilder.load_data()
        
        # Check that data was loaded
        self.assertGreater(len(queries_data['queries']), 0)
        self.assertGreater(len(documents_data['documents']), 0)
        self.assertGreater(len(training_data), 0)
    
    def test_build_feature_matrix(self):
        """Test building feature matrix."""
        queries_data, documents_data, training_data = DatasetBuilder.load_data()
        documents = documents_data['documents']
        
        extractor = FeatureExtractor(documents)
        features, relevances, query_ids = DatasetBuilder.build_feature_matrix(
            training_data, queries_data, extractor
        )
        
        # Check shapes
        self.assertEqual(features.shape[0], len(relevances))
        self.assertEqual(features.shape[0], len(query_ids))
        
        # Check that we have multiple queries
        self.assertGreater(len(np.unique(query_ids)), 1)


if __name__ == '__main__':
    print("Running unit tests...")
    unittest.main(verbosity=2)
