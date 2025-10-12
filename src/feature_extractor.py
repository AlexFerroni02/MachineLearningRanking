"""
Feature Extractor Module

Extracts features from query-document pairs for ranking.
"""

import re
import math
from typing import Dict, List
from collections import Counter
import numpy as np


class FeatureExtractor:
    """Extracts ranking features from query-document pairs."""
    
    def __init__(self, documents: List[Dict]):
        """Initialize feature extractor with document corpus.
        
        Args:
            documents: List of document dictionaries
        """
        self.documents = documents
        self.doc_lookup = {doc['doc_id']: doc for doc in documents}
        
        # Precompute IDF scores
        self.idf_scores = self._compute_idf()
        
        # Average document length for BM25
        self.avg_doc_len = np.mean([len(self._tokenize(doc['content'])) 
                                     for doc in documents])
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _compute_idf(self) -> Dict[str, float]:
        """Compute IDF scores for all terms in corpus.
        
        Returns:
            Dictionary mapping terms to IDF scores
        """
        num_docs = len(self.documents)
        doc_freq = Counter()
        
        # Count document frequency for each term
        for doc in self.documents:
            tokens = set(self._tokenize(doc['content'] + ' ' + doc['title']))
            for token in tokens:
                doc_freq[token] += 1
        
        # Compute IDF: log(N / df)
        idf = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((num_docs + 1) / (df + 1))
        
        return idf
    
    def _compute_tf(self, text: str) -> Dict[str, int]:
        """Compute term frequency for text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping terms to frequency counts
        """
        tokens = self._tokenize(text)
        return Counter(tokens)
    
    def _compute_tfidf_score(self, query_tokens: List[str], doc_text: str) -> float:
        """Compute TF-IDF score between query and document.
        
        Args:
            query_tokens: List of query tokens
            doc_text: Document text
            
        Returns:
            TF-IDF similarity score
        """
        doc_tf = self._compute_tf(doc_text)
        score = 0.0
        
        for term in query_tokens:
            tf = doc_tf.get(term, 0)
            idf = self.idf_scores.get(term, 0)
            score += tf * idf
        
        return score
    
    def _compute_bm25_score(self, query_tokens: List[str], doc_text: str, 
                           k1: float = 1.5, b: float = 0.75) -> float:
        """Compute BM25 score between query and document.
        
        Args:
            query_tokens: List of query tokens
            doc_text: Document text
            k1: BM25 parameter (term saturation)
            b: BM25 parameter (length normalization)
            
        Returns:
            BM25 score
        """
        doc_tokens = self._tokenize(doc_text)
        doc_len = len(doc_tokens)
        doc_tf = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term in doc_tf:
                tf = doc_tf[term]
                idf = self.idf_scores.get(term, 0)
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
        
        return score
    
    def _exact_match_count(self, query_tokens: List[str], doc_text: str) -> int:
        """Count exact matches of query terms in document.
        
        Args:
            query_tokens: List of query tokens
            doc_text: Document text
            
        Returns:
            Number of exact matches
        """
        doc_tokens = self._tokenize(doc_text)
        doc_token_set = set(doc_tokens)
        return sum(1 for term in query_tokens if term in doc_token_set)
    
    def _cosine_similarity(self, query_tokens: List[str], doc_text: str) -> float:
        """Compute cosine similarity between query and document.
        
        Args:
            query_tokens: List of query tokens
            doc_text: Document text
            
        Returns:
            Cosine similarity score
        """
        doc_tokens = self._tokenize(doc_text)
        query_tf = Counter(query_tokens)
        doc_tf = Counter(doc_tokens)
        
        # Get all unique terms
        all_terms = set(query_tokens) | set(doc_tokens)
        
        # Create vectors
        query_vec = np.array([query_tf.get(term, 0) for term in all_terms])
        doc_vec = np.array([doc_tf.get(term, 0) for term in all_terms])
        
        # Compute cosine similarity
        dot_product = np.dot(query_vec, doc_vec)
        query_norm = np.linalg.norm(query_vec)
        doc_norm = np.linalg.norm(doc_vec)
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return dot_product / (query_norm * doc_norm)
    
    def _term_overlap_ratio(self, query_tokens: List[str], doc_text: str) -> float:
        """Compute ratio of query terms appearing in document.
        
        Args:
            query_tokens: List of query tokens
            doc_text: Document text
            
        Returns:
            Overlap ratio (0 to 1)
        """
        if not query_tokens:
            return 0.0
        
        doc_tokens = set(self._tokenize(doc_text))
        overlap = sum(1 for term in query_tokens if term in doc_tokens)
        return overlap / len(query_tokens)
    
    def extract_features(self, query: str, doc_id: str) -> np.ndarray:
        """Extract all features for a query-document pair.
        
        Args:
            query: Query text
            doc_id: Document ID
            
        Returns:
            Numpy array of feature values
        """
        doc = self.doc_lookup.get(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        
        query_tokens = self._tokenize(query)
        doc_content = doc['content']
        doc_title = doc['title']
        combined_text = doc_title + ' ' + doc_content
        
        features = []
        
        # Text similarity features
        features.append(self._compute_tfidf_score(query_tokens, combined_text))
        features.append(self._compute_tfidf_score(query_tokens, doc_title))
        features.append(self._compute_bm25_score(query_tokens, combined_text))
        features.append(self._compute_bm25_score(query_tokens, doc_title))
        
        # Exact match features
        features.append(self._exact_match_count(query_tokens, combined_text))
        features.append(self._exact_match_count(query_tokens, doc_title))
        
        # Semantic features
        features.append(self._cosine_similarity(query_tokens, combined_text))
        features.append(self._term_overlap_ratio(query_tokens, combined_text))
        
        # Document length features
        features.append(len(self._tokenize(doc_content)))
        features.append(len(self._tokenize(doc_title)))
        
        # Query term frequency in document
        doc_tf = self._compute_tf(combined_text)
        avg_term_freq = np.mean([doc_tf.get(term, 0) for term in query_tokens])
        features.append(avg_term_freq)
        
        # Binary features for metadata
        features.append(1.0 if doc.get('clinical_relevance') == 'high' else 0.0)
        features.append(1.0 if doc.get('test_type') == 'quantitative' else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features.
        
        Returns:
            List of feature names
        """
        return [
            'tfidf_content',
            'tfidf_title',
            'bm25_content',
            'bm25_title',
            'exact_match_content',
            'exact_match_title',
            'cosine_similarity',
            'term_overlap_ratio',
            'doc_length',
            'title_length',
            'avg_term_freq',
            'high_clinical_relevance',
            'quantitative_test'
        ]


if __name__ == '__main__':
    # Example usage
    import json
    from pathlib import Path
    
    # Load documents
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "documents.json", 'r') as f:
        docs_data = json.load(f)
        documents = docs_data['documents']
    
    # Initialize feature extractor
    extractor = FeatureExtractor(documents)
    
    # Extract features for a query-document pair
    query = "glucose in blood"
    doc_id = "D1"
    
    features = extractor.extract_features(query, doc_id)
    feature_names = extractor.get_feature_names()
    
    print("Feature Extraction Demo")
    print("=" * 50)
    print(f"Query: {query}")
    print(f"Document: {doc_id}")
    print("\nFeatures:")
    for name, value in zip(feature_names, features):
        print(f"  {name:25s}: {value:.4f}")
