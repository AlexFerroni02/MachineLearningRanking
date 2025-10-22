# MachineLearningRanking

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AlexFerroni02/Binary-Classifier-for-Nutritional-IR.git
   cd Binary-Classifier-for-Nutritional-IR
    ```
2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `.\venv\Scripts\activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
   
# Machine-Learned Ranking Pipeline

This project implements the data preparation pipeline for a Learning to Rank (LTR) model based on the concepts presented in the "Machine Learned Ranking" slides and the methodology from Joachims' paper on Ranking SVM.

The goal is to transform a dataset of documents and a set of queries into a training set suitable for a pairwise LTR model like Ranking SVM.

## Prerequisites

Ensure you have Python installed with the following libraries:
- pandas
- scikit-learn
- numpy

You can install them via pip:
~~~bash
pip install pandas scikit-learn numpy
~~~

## Data Structure

Your initial data should be placed in a `data/` directory with the following structure:

~~~
project-folder/
│
├── data/
│   ├── loinc_dataset-v2.xlsx   # Multi-sheet Excel file with documents for each query.
│   └── queries.xlsx            # Simple Excel file listing the queries.
│
├── scripts/
│   ├── generate_create_feature_and_ranking.py
│   ├── simula_clicks.py
│   ├── crea_coppie_preferenza.py
│   └── trasforma_per_svm.py
│
└── README.md
~~~

-----

## The Data Preparation Pipeline

The pipeline consists of four sequential steps. Each step is performed by a dedicated Python script.

### Step 1: Feature Engineering & Baseline Ranking

This step reads the raw document data, cleans it, extracts a rich set of features, and creates an initial baseline ranking for each query.

**Script**: `generate_feature_ranking.py`

**What it does**:

1.  Reads each sheet from `expanded_loinc_dataset.xlsx`, treating each as a separate query-document set.
2.  **Normalizes** specialized LOINC codes in the `system` and `property` fields (e.g., `Bld` becomes `blood`). This normalization is done temporarily for calculation purposes, leaving the original data intact in the final output.
3.  Extracts a wide range of **numerical features** for each document relative to its query, including:
    * TF-IDF cosine similarity for *each* major text field (`long_common_name`, `component`, etc.).
    * A count of how many query terms appear in each field.
4.  Computes a **holistic baseline ranking score** (`ranking_score_baseline`) by calculating the cosine similarity between the query and a combined text field (`component` + `system` + `property`).
5.  Creates the baseline ranking by sorting documents based on this `ranking_score_baseline`.
6.  Saves the results into separate CSV files, one for each query.
**How to run it**:

~~~bash
python Pre_process/generate_feature_ranking.py
~~~

**Output**:

  - A new directory: `data/feature_rankings/`
  - Inside, you will find a CSV file for each query (e.g., `glucose_in_blood.csv`), containing all the original document data enriched with the newly calculated feature columns.

-----

### Step 2: Simulating User Clicks

This script takes the ranked lists from the previous step and simulates user click behavior to generate training signals.

**Script**: `simulate_clicks.py`

**What it does**:

1.  Reads every ranked CSV file from the `data/feature_rankings/` directory.
2.  Applies an **advanced cascade model** to simulate clicks. This model is more realistic than simple probabilistic models because it assumes:
      * A user scans results from top to bottom.
      * The probability of a click depends on both the document's **rank** and its **relevance** (approximated by its similarity score).
      * An user may stop searching after a satisfying click.
3.  Adds a new boolean column named `clicked` to each DataFrame.
4.  Saves the results into a new set of CSV files.

**How to run it**:

~~~bash
python Pre_processing/simulate_clicks.py
~~~

**Output**:

  - A new directory: `data/click_logs/`
  - Inside, you will find CSV files with the same names as before, but now including the `clicked` column.

-----

### Step 3: Generating Conceptual Preference Pairs

This script implements the core logic from Joachims' paper by converting click data into human-readable preference pairs.

**Script**: `create_pair.py`

**What it does**:

1.  Reads every file from the `data/click_logs/` directory.
2.  For each query, it identifies all clicked documents.
3.  [cite_start]Based on the rule **"if a user skips document `j` to click a lower-ranked document `i`, then `i` is preferred over `j`"** [cite: 363-365], it generates preference pairs.
4.  Consolidates all pairs from all queries into a single, human-readable CSV file.

**How to run it**:

~~~bash
python Pre_processing/create_pair.py
~~~

**Output**:

  - A new file: `data/conceptual_preference_pairs.csv`
  - This file has four columns: `query`, `preferred_doc_id`, `not_preferred_doc_id`, and `label`. Each row represents a single preference judgment.

-----

### Step 4: Transforming Pairs into SVM Training Data

This is the final transformation step. It converts the conceptual preference pairs into a numerical format that the SVM algorithm can process.

**Script**: `transform_for_svm.py`

**What it does**:

1.  Reads the `conceptual_preference_pairs.csv` file.
2.  For each preference pair (`doc_i > doc_j`), it retrieves the corresponding numerical feature vectors for both documents from the `data/feature_rankings/` files.
3.  It then calculates the **difference vector** (`feature_vector_i - feature_vector_j`). [cite_start]This transformation turns the ranking problem into a binary classification problem, which is the key insight of the Ranking SVM method [cite: 466-468].
4.  Saves these difference vectors into a final, consolidated training dataset. For each positive pair, an inverse negative pair is also added to create a balanced training set.

**How to run it**:

~~~bash
python scripts/transform_for_svm.py
~~~

**Output**:

  - The final training file: `data/svm_training_dataset.csv`
  - Each row in this file is a training example for the SVM. The columns represent the differences in feature values (`feature_name_diff`), and the final column is the `label` (+1 or -1).

-----

## Training the SVM Model

This script takes the conceptual pairs and raw features, performs the SVM transformation, and trains the final model.

**Script**: `train.py`

**What it does**:

1.  Loads the `conceptual_preference_pairs.csv` and *all* document features from the `data/feature_rankings/` directory.
2.  Splits the *preference pairs* into a training and test set (e.g., 80/20). This is a crucial step to prevent data leakage.
3.  Initializes a `StandardScaler` and trains it (`.fit()`) *only* on the features of documents that appear in the *training pairs*.
4.  Generates the final training and test datasets by creating **scaled difference vectors**.
    * For each pair (`doc_i > doc_j`), it computes `(scaled_vec_i - scaled_vec_j)` and assigns the label `+1`.
    * It also adds the inverse pair `(scaled_vec_j - scaled_vec_i)` with the label `-1` to create a balanced binary classification problem.
5.  Performs a `GridSearchCV` to find the optimal `C` hyperparameter for a linear SVM (`SVC(kernel='linear')`).
6.  Evaluates the best model on the held-out test set to report final accuracy.
7.  Saves the final trained `model`, the fitted `scaler`, and a JSON file of the raw `feature_names` to the `model/` directory.

**How to run it**:

~~~bash
python train.py
~~~

**Output**:

* A new directory: `model/`
* `model/ranking_svm_model_optimized.joblib`: The trained SVM model.
* `model/feature_scaler.joblib`: The scaler fitted to the training data.
* `model/feature_names.json`: A list of the raw feature names used for training.

-----

## Applying the Model for Ranking

This script demonstrates how to use the trained model and scaler to rank a new, unseen set of documents for a query.

**Script**: `ranker.py`

**What it does**:

1.  Loads the trained `model`, the fitted `scaler`, and the `feature_names` from the `model/` directory.
2.  Loads a CSV file containing documents and their pre-computed features (e.g., an output file from Step 1).
3.  Applies the loaded `scaler` to transform the document features (`scaler.transform`).
4.  Calculates the final `ranking_score` for each document. This is done by computing the dot product of the **scaled feature vector** and the model's learned **weight vector** (`w = model.coef_[0]`).
5.  Prints the final list of documents, sorted from highest to lowest `ranking_score`.

**How to run it**:

~~~bash
python ranker.py
~~~

**Output**:

* A ranked list of documents printed to the console, showing the top-scoring documents as determined by the learned LTR model.