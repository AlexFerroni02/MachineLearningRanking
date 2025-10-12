# MachineLearningRanking

A Machine Learning to Rank (MLR) system for medical laboratory test search, focusing on educational understanding of ranking approaches.

## Overview

This project implements a **Learning to Rank (LTR)** system using a **Listwise approach with LambdaMART** algorithm. The system is designed to rank medical laboratory test documents based on their relevance to user queries.

## Chosen MLR Approach: LambdaMART (Listwise Ranking)

### What is LambdaMART?

LambdaMART is a state-of-the-art listwise Learning to Rank algorithm that combines:
- **LambdaRank**: A neural network approach that directly optimizes ranking metrics (NDCG, MAP)
- **MART (Multiple Additive Regression Trees)**: Gradient boosted decision trees

### How It Works

1. **Training Phase**:
   - Takes query-document pairs with relevance labels
   - Extracts features from documents (TF-IDF, BM25, exact matches, etc.)
   - Uses gradient boosting to build an ensemble of decision trees
   - Optimizes directly for ranking metrics (NDCG) using lambda gradients

2. **Ranking Phase**:
   - Given a query, extracts features for candidate documents
   - Passes features through the trained model
   - Outputs relevance scores for each document
   - Sorts documents by score to produce final ranking

3. **Why LambdaMART?**:
   - Considers the entire ranked list (listwise) rather than individual pairs
   - Directly optimizes for ranking metrics (NDCG, MAP)
   - Robust and widely used in production systems (Bing, etc.)
   - Better performance than pointwise or pairwise approaches

## Project Structure

```
MachineLearningRanking/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── data/                       # Dataset directory
│   ├── loinc_mapping.json     # Lab test to LOINC code mapping
│   ├── documents.json         # Medical document corpus
│   ├── queries.json           # Search queries
│   └── training_data.json     # Query-doc pairs with relevance labels
├── src/                        # Source code
│   ├── __init__.py
│   ├── loinc_mapper.py        # LOINC mapping utilities
│   ├── dataset_builder.py     # Training set creation
│   ├── feature_extractor.py   # Feature engineering
│   ├── ranker.py              # MLR model implementation
│   └── evaluation.py          # Metrics (NDCG, MAP, MRR)
├── scripts/                    # Executable scripts
│   ├── train.py               # Train the ranking model
│   └── demo.py                # Interactive demo
└── tests/                      # Unit tests
    └── test_ranker.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python scripts/train.py
```

### 2. Run Demo

```bash
python scripts/demo.py
```

### 3. Evaluate Model

The training script automatically evaluates using NDCG@k, MAP, and MRR metrics.

## Dataset

The system includes:

### Initial Queries (3 core medical queries):
1. "glucose in blood" - Blood glucose level testing
2. "bilirubin in plasma" - Plasma bilirubin measurement
3. "white blood cells count" - WBC enumeration

### Extended Queries (7 additional queries):
4. "hemoglobin measurement"
5. "creatinine in serum"
6. "cholesterol total"
7. "platelet count"
8. "liver function tests"
9. "kidney function panel"
10. "complete blood count"

### Documents
- Medical test descriptions with LOINC codes
- Clinical documentation
- Test procedure descriptions
- Extended with additional medical terminology

## LOINC Mapping

LOINC (Logical Observation Identifiers Names and Codes) is a universal standard for identifying medical laboratory observations. This project maps common lab tests to their LOINC codes for standardization.

Example mappings:
- Glucose in Blood → LOINC: 2345-7
- Bilirubin in Plasma → LOINC: 1975-2
- White Blood Cells Count → LOINC: 6690-2

## Features Used

The ranking model uses the following features:

1. **Text-based Features**:
   - TF-IDF scores
   - BM25 scores
   - Exact term matches
   - Partial term matches

2. **Metadata Features**:
   - Document length
   - Term frequency in document
   - Document type indicators

3. **Semantic Features**:
   - Query-document similarity (cosine)
   - Term overlap ratio

## Evaluation Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain (primary metric)
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank

## Learning Objectives

This implementation demonstrates:
1. ✓ Understanding of Learning to Rank approaches
2. ✓ Listwise ranking with LambdaMART
3. ✓ LOINC mapping for medical terminology standardization
4. ✓ Training set construction for ranking tasks
5. ✓ Feature engineering for text ranking
6. ✓ Model implementation using LightGBM
7. ✓ Dataset extension (terms and queries)
8. ✓ Evaluation using standard IR metrics

## References

- LambdaMART: "From RankNet to LambdaRank to LambdaMART" by Burges (2010)
- LOINC: https://loinc.org/
- LightGBM: https://lightgbm.readthedocs.io/

## License

MIT License