# Quick Start Guide

This guide helps you get started with the Machine Learning Ranking system.

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/AlexFerroni02/MachineLearningRanking.git
cd MachineLearningRanking
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Running the System

### Option 1: Use Pre-trained Model (Demo)

Run the interactive demo to search for medical tests:

```bash
python scripts/demo.py
```

Example queries to try:
- "glucose in blood"
- "liver function tests"
- "complete blood count"
- "kidney function"

### Option 2: Train Your Own Model

Train a new ranking model from scratch:

```bash
python scripts/train.py
```

This will:
1. Load the dataset (queries, documents, relevance judgments)
2. Extract features for all query-document pairs
3. Train the LambdaMART model
4. Evaluate performance with NDCG, MAP, MRR metrics
5. Save the trained model to `model/lambdamart_ranker.txt`

Training takes approximately 1-2 minutes.

### Option 3: Run Unit Tests

Verify the implementation with unit tests:

```bash
python tests/test_ranker.py
```

## Using Individual Components

### LOINC Mapper

Map laboratory tests to standard LOINC codes:

```python
from src.loinc_mapper import LOINCMapper

mapper = LOINCMapper()
code = mapper.get_loinc_code('glucose_in_blood')
print(f"LOINC code: {code}")  # Output: 2345-7

# Search by keyword
results = mapper.search_by_keyword('blood')
for result in results:
    print(f"{result['test_name']}: {result['loinc_code']}")
```

### Feature Extractor

Extract ranking features from query-document pairs:

```python
import json
from src.feature_extractor import FeatureExtractor

# Load documents
with open('data/documents.json', 'r') as f:
    docs_data = json.load(f)
    documents = docs_data['documents']

# Create extractor
extractor = FeatureExtractor(documents)

# Extract features
query = "glucose in blood"
doc_id = "D1"
features = extractor.extract_features(query, doc_id)

print(f"Features: {features}")
print(f"Feature names: {extractor.get_feature_names()}")
```

### Ranking Model

Use the trained model to rank documents:

```python
import numpy as np
from src.ranker import LambdaMARTRanker, DatasetBuilder
from src.feature_extractor import FeatureExtractor

# Load data
queries_data, documents_data, _ = DatasetBuilder.load_data()
documents = documents_data['documents']

# Load model
ranker = LambdaMARTRanker()
ranker.load_model('model/lambdamart_ranker.txt')

# Create feature extractor
extractor = FeatureExtractor(documents)

# Rank documents for a query
query = "kidney function"
features_list = [extractor.extract_features(query, doc['doc_id']) 
                 for doc in documents]
X = np.array(features_list)
scores = ranker.predict(X)

# Get top 5
top_indices = np.argsort(scores)[::-1][:5]
for i, idx in enumerate(top_indices, 1):
    print(f"{i}. {documents[idx]['title']} (score: {scores[idx]:.4f})")
```

### Evaluation Metrics

Compute ranking metrics:

```python
from src.evaluation import RankingMetrics

# Example: relevances in ranked order
relevances = [3, 2, 1, 0, 0]

# Compute metrics
ndcg_5 = RankingMetrics.ndcg(relevances, k=5)
map_score = RankingMetrics.average_precision(relevances)
mrr = RankingMetrics.reciprocal_rank(relevances)

print(f"NDCG@5: {ndcg_5:.4f}")
print(f"MAP: {map_score:.4f}")
print(f"MRR: {mrr:.4f}")
```

## Dataset Structure

### Queries (`data/queries.json`)
- 10 queries about medical laboratory tests
- Fields: qid, text, intent, category

### Documents (`data/documents.json`)
- 20 medical test descriptions
- Fields: doc_id, title, content, loinc_code, test_type, sample_type, clinical_relevance

### Training Data (`data/training_data.json`)
- 80 query-document pairs with relevance labels
- Relevance scale: 0 (not relevant) to 3 (perfect match)

### LOINC Mapping (`data/loinc_mapping.json`)
- 10 laboratory tests mapped to LOINC codes
- Includes synonyms and metadata

## Understanding the Results

### NDCG (Normalized Discounted Cumulative Gain)
- Range: 0.0 to 1.0 (higher is better)
- Measures ranking quality with position-based discounting
- NDCG@10 = 0.92 means excellent ranking performance

### MAP (Mean Average Precision)
- Range: 0.0 to 1.0 (higher is better)
- Average precision across all queries
- MAP = 0.87 indicates high precision

### MRR (Mean Reciprocal Rank)
- Range: 0.0 to 1.0 (higher is better)
- Average of 1/rank of first relevant document
- MRR = 0.95 means first relevant doc is usually in position 1-2

### Precision@k
- Proportion of relevant documents in top k results
- P@3 = 0.83 means 83% of top-3 results are relevant

## Customization

### Adding New Queries

Edit `data/queries.json`:
```json
{
  "qid": "Q11",
  "text": "your new query",
  "intent": "description of intent",
  "category": "category_name"
}
```

### Adding New Documents

Edit `data/documents.json`:
```json
{
  "doc_id": "D21",
  "title": "Document Title",
  "content": "Document content...",
  "loinc_code": "XXXX-X",
  "test_type": "quantitative",
  "sample_type": "blood",
  "clinical_relevance": "high"
}
```

### Adding Relevance Judgments

Edit `data/training_data.json`:
```json
{
  "qid": "Q11",
  "doc_id": "D21",
  "relevance": 3,
  "label": "perfect_match"
}
```

Then retrain the model:
```bash
python scripts/train.py
```

## Troubleshooting

### ImportError: No module named 'lightgbm'
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Model file not found
**Solution**: Train the model first:
```bash
python scripts/train.py
```

### Low ranking performance
**Solution**: 
- Add more training data
- Adjust relevance labels
- Add more features
- Tune model parameters in `src/ranker.py`

## Further Reading

- **APPROACH.md**: Detailed explanation of the LambdaMART algorithm
- **README.md**: Project overview and architecture
- **Source code**: Well-commented code in `src/` directory

## Support

For questions or issues, please refer to:
- APPROACH.md for understanding the algorithm
- Source code documentation
- Unit tests in `tests/` for usage examples
