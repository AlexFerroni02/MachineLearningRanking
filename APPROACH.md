# Machine Learning to Rank (MLR) - Detailed Explanation

## Understanding the Approach

This document provides a comprehensive explanation of the Machine Learning to Rank approach implemented in this project, fulfilling the requirement to "Understand and be able to Explain the process."

## 1. Chosen MLR Approach: LambdaMART

### What is LambdaMART?

LambdaMART is a **listwise Learning to Rank** algorithm that represents the state-of-the-art in ranking systems. It combines two powerful concepts:

1. **LambdaRank**: A neural network-based ranking algorithm that uses lambda gradients
2. **MART (Multiple Additive Regression Trees)**: Gradient boosted decision trees

### Why Listwise Instead of Pointwise or Pairwise?

There are three main approaches to Learning to Rank:

#### Pointwise Approach
- **How it works**: Treats ranking as a regression/classification problem
- **Prediction**: Assigns a relevance score to each document independently
- **Limitation**: Doesn't consider the relative order of documents
- **Example**: Predicting relevance scores 0-3 for each document

#### Pairwise Approach  
- **How it works**: Learns to compare pairs of documents
- **Prediction**: Determines which document should rank higher in each pair
- **Limitation**: Only considers local pairwise preferences, not global list quality
- **Example**: RankNet, RankSVM

#### Listwise Approach (Our Choice - LambdaMART)
- **How it works**: Optimizes for the entire ranked list quality
- **Prediction**: Directly optimizes ranking metrics like NDCG
- **Advantage**: Considers the entire ranking context and optimizes what we actually care about
- **Result**: Better performance on ranking metrics

### LambdaMART Algorithm Details

#### Training Phase

1. **Input Data**:
   - Query-document pairs with relevance labels (0 = not relevant, 1 = related, 2 = excellent, 3 = perfect match)
   - Features extracted from each query-document pair
   - Query groups (which documents belong to which query)

2. **Lambda Gradients**:
   - Instead of traditional gradients, LambdaMART uses "lambda" gradients
   - Lambda gradients are computed based on the change in NDCG when swapping documents
   - This directly optimizes for the ranking metric we care about

3. **Gradient Boosting**:
   - Builds an ensemble of regression trees iteratively
   - Each tree corrects the errors of the previous trees
   - Uses lambda gradients instead of traditional loss gradients

4. **Tree Building**:
   - At each iteration, builds a regression tree
   - Leaf values are optimized to maximize NDCG
   - Trees are added to the ensemble with a learning rate

#### Ranking Phase

1. **Feature Extraction**: Extract features for all candidate documents
2. **Score Prediction**: Pass features through the trained ensemble
3. **Ranking**: Sort documents by predicted scores (descending)

### Mathematical Foundation

The key insight is the lambda gradient:

```
λᵢⱼ = -∂NDCG/∂sᵢⱼ × |ΔNDCGᵢⱼ|
```

Where:
- `λᵢⱼ` is the lambda gradient for swapping documents i and j
- `sᵢⱼ` is the score difference between documents i and j
- `ΔNDCGᵢⱼ` is the change in NDCG when swapping i and j

This means the gradient is weighted by how much the ranking metric would improve if we got this pair correct.

## 2. Feature Engineering

### Features Used (13 total)

Our system extracts 13 features for each query-document pair:

#### Text Similarity Features (4 features)
1. **TF-IDF Content Score**: Term frequency-inverse document frequency for document content
2. **TF-IDF Title Score**: TF-IDF for document title
3. **BM25 Content Score**: Okapi BM25 ranking function for content
4. **BM25 Title Score**: BM25 for title

#### Exact Match Features (2 features)
5. **Exact Match Content**: Number of query terms appearing in content
6. **Exact Match Title**: Number of query terms appearing in title

#### Semantic Features (2 features)
7. **Cosine Similarity**: Vector space model similarity
8. **Term Overlap Ratio**: Proportion of query terms in document

#### Document Statistics (2 features)
9. **Document Length**: Number of tokens in content
10. **Title Length**: Number of tokens in title

#### Query-Document Features (1 feature)
11. **Average Term Frequency**: Mean frequency of query terms in document

#### Metadata Features (2 features)
12. **High Clinical Relevance**: Binary indicator (1 if high relevance, 0 otherwise)
13. **Quantitative Test**: Binary indicator (1 if quantitative test, 0 otherwise)

### Why These Features?

- **TF-IDF and BM25**: Classic IR features proven effective for text matching
- **Exact matches**: Medical terminology often requires exact matching
- **Semantic features**: Capture broader relevance beyond exact matches
- **Document statistics**: Longer documents may contain more information
- **Metadata**: Domain-specific features for medical tests

## 3. Training Set Construction

### Data Structure

The training set consists of:
- **10 queries** (3 core + 7 extended)
- **20 documents** (medical test descriptions)
- **80 query-document pairs** with relevance judgments

### Relevance Labels

We use a 4-level graded relevance scale:
- **3 (Perfect Match)**: Document directly answers the query
- **2 (Excellent)**: Highly relevant, provides useful information
- **1 (Related)**: Mentions related concepts but not the main topic
- **0 (Not Relevant)**: Unrelated to the query

### Example Training Instance

For query "glucose in blood" and document "Blood Glucose Testing":
- Query ID: Q1
- Document ID: D1
- Relevance: 3 (perfect match)
- Features: [13.68, 5.08, 5.08, 5.34, 3, 3, 0.69, 1.0, 56, 8, 4.67, 1, 1]

## 4. Model Training Process

### Step-by-Step Training

1. **Data Loading**: Load queries, documents, and relevance judgments
2. **Feature Extraction**: Extract 13 features for each query-document pair
3. **Data Preparation**: Create feature matrix, group by queries
4. **Model Training**: 
   - Use LightGBM with lambdarank objective
   - Optimize for NDCG metric
   - 100 boosting rounds with early stopping
5. **Validation**: Monitor NDCG on validation set
6. **Model Saving**: Save trained model for inference

### LightGBM Parameters

```python
{
    'objective': 'lambdarank',      # LambdaMART algorithm
    'metric': 'ndcg',                # Optimize NDCG
    'ndcg_eval_at': [1, 3, 5, 10],  # Evaluate at multiple cutoffs
    'num_leaves': 31,                # Tree complexity
    'learning_rate': 0.05,           # Step size
    'feature_fraction': 0.9,         # Feature sampling
    'bagging_fraction': 0.8,         # Data sampling
    'max_depth': 6                   # Maximum tree depth
}
```

## 5. Evaluation Metrics

### Primary Metric: NDCG (Normalized Discounted Cumulative Gain)

NDCG measures ranking quality with position-based discounting:

```
DCG@k = Σᵢ₌₁ᵏ (2^{rel_i} - 1) / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k
```

Where:
- `rel_i` is the relevance of the document at position i
- `IDCG` is the ideal DCG (best possible ranking)
- Higher positions have more impact (logarithmic discount)

**Our Results**:
- NDCG@1: 0.7667
- NDCG@3: 0.8578
- NDCG@10: 0.9175

### Secondary Metrics

#### Mean Average Precision (MAP)
- Precision at each relevant document position
- Averaged over all queries
- **Our Result**: 0.8677

#### Mean Reciprocal Rank (MRR)
- Reciprocal of the rank of the first relevant document
- **Our Result**: 0.9500 (first relevant document typically at position 1-2)

#### Precision@k
- Proportion of relevant documents in top k results
- **Our Result (P@3)**: 0.8333

## 6. Dataset Extensions

### Extension 1: More Terms (Medical Terminology)

We extended the dataset with additional medical terminology:
- Added 10 more documents (total 20)
- Included panel tests (CBC, Liver Function Panel, Kidney Function Panel)
- Added related conditions (diabetes, anemia, liver disease)
- Incorporated medical abbreviations (ALT, AST, BUN, Hgb, WBC)

### Extension 2: More Queries

Extended from 3 core queries to 10 total queries:

**Core Queries (3)**:
1. glucose in blood
2. bilirubin in plasma
3. white blood cells count

**Extended Queries (7)**:
4. hemoglobin measurement
5. creatinine in serum
6. cholesterol total
7. platelet count
8. liver function tests
9. kidney function panel
10. complete blood count

## 7. LOINC Mapping (Independent Subtask)

### What is LOINC?

LOINC (Logical Observation Identifiers Names and Codes) is a universal standard for:
- Identifying medical laboratory observations
- Clinical measurements
- Diagnostic studies

### Our LOINC Implementation

We mapped 10 common laboratory tests to their LOINC codes:

| Test | LOINC Code | Long Name |
|------|------------|-----------|
| Glucose in Blood | 2345-7 | Glucose [Mass/volume] in Blood |
| Bilirubin in Plasma | 1975-2 | Bilirubin.total [Mass/volume] in Serum or Plasma |
| White Blood Cells Count | 6690-2 | Leukocytes [#/volume] in Blood by Automated count |
| Hemoglobin | 718-7 | Hemoglobin [Mass/volume] in Blood |
| Creatinine in Serum | 2160-0 | Creatinine [Mass/volume] in Serum or Plasma |

### LOINC Structure

Each LOINC code has 6 parts:
1. **Component**: What is measured (e.g., Glucose)
2. **Property**: Type of measurement (e.g., Mass concentration)
3. **Timing**: When measured (e.g., Point in time)
4. **System**: Where measured (e.g., Blood)
5. **Scale**: How measured (e.g., Quantitative)
6. **Method**: Measurement method (optional)

## 8. Key Advantages of This Approach

1. **Optimizes for What Matters**: Directly optimizes NDCG, not a surrogate loss
2. **Handles Graded Relevance**: Works with multiple relevance levels (0-3)
3. **Considers Full Context**: Listwise approach considers entire ranking
4. **Production-Ready**: LambdaMART is used by major search engines (Bing)
5. **Interpretable**: Feature importance shows what drives rankings
6. **Extensible**: Easy to add new features and data

## 9. Limitations and Future Work

### Current Limitations
- Small dataset (10 queries, 20 documents)
- Manual relevance judgments
- Simple features (no deep learning embeddings)

### Potential Improvements
- Add semantic embeddings (BERT, BioBERT)
- Collect more queries and documents
- Implement click-through data learning
- Add query expansion and synonyms
- Integrate with medical knowledge graphs

## 10. Conclusion

This implementation demonstrates a complete Learning to Rank system:
- ✅ Chosen approach: LambdaMART (listwise ranking)
- ✅ Understanding: Explained algorithm and mathematical foundation
- ✅ LOINC mapping: Mapped lab tests to standard codes
- ✅ Training set: Built for 10 queries with graded relevance
- ✅ Implementation: Working model using LightGBM
- ✅ Dataset extension: Extended terms and queries
- ✅ Evaluation: NDCG, MAP, MRR metrics

The system achieves strong performance (NDCG@10: 0.92) and provides a solid foundation for medical laboratory test search and ranking.
