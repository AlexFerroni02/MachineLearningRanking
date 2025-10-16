import pandas as pd
import joblib
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_FILE = "ranking_svm_model_optimized.joblib"
EXAMPLE_FEATURE_FILE = "../data/feature_rankings/glucose_in_blood.csv"


def rank_documents(model, documents_df):
    """
    Uses a trained SVM model to calculate ranking scores for a set of documents.
    """
    print("\n--- Ranking Documents with Trained Model ---")

    # Identify the feature columns the model expects
    feature_cols = [col.replace('_diff', '') for col in model.feature_names_in_]

    # Ensure all required feature columns are present in the input DataFrame
    if not all(col in documents_df.columns for col in feature_cols):
        print("ERROR: The document file is missing one or more required feature columns.")
        return None

    # Select ONLY the numeric feature columns for calculation
    doc_features = documents_df[feature_cols]

    # Extract the learned weight vector 'w'
    w = model.coef_[0]

    # Calculate the ranking score for each document
    ranking_scores = np.dot(doc_features.values, w)

    # Add scores and sort to get the final ranking
    ranked_df = documents_df.copy()
    ranked_df['ranking_score'] = ranking_scores
    ranked_df = ranked_df.sort_values(by='ranking_score', ascending=False).reset_index(drop=True)

    return ranked_df


def main():
    """
    Loads a pre-trained model and uses it to rank documents from a file.
    """
    # 1. Load the pre-trained model
    try:
        model = joblib.load(MODEL_FILE)
        print(f"Successfully loaded model from '{MODEL_FILE}'")
    except FileNotFoundError:
        print(f"ERROR: Model file '{MODEL_FILE}' not found. Please run the training script first.")
        return

    # 2. Load the documents you want to rank
    try:
        docs_to_rank_df = pd.read_csv(EXAMPLE_FEATURE_FILE)
        print(f"Loaded {len(docs_to_rank_df)} documents to rank from '{EXAMPLE_FEATURE_FILE}'")
    except FileNotFoundError:
        print(f"ERROR: Document feature file '{EXAMPLE_FEATURE_FILE}' not found.")
        return

    # 3. Get the new ranking
    final_ranking = rank_documents(model, docs_to_rank_df)

    if final_ranking is not None:
        print("\nTop 5 ranked documents based on the learned model:")
        print(final_ranking[['loinc_num', 'long_common_name', 'ranking_score']].head(5))


if __name__ == "__main__":
    main()