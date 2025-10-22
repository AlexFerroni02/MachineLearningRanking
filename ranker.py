import pandas as pd
import joblib
import numpy as np
import os
import json

# --- CONFIGURATION ---
MODEL_FILE = "./model/ranking_svm_model_optimized.joblib"
SCALER_FILE = "./model/feature_scaler.joblib"
FEATURE_NAMES_FILE = "./model/feature_names.json"
EXAMPLE_FEATURE_FILE = "data/feature_rankings/glucose_in_blood.csv"


def rank_documents(model, scaler, documents_df, feature_names):
    """Uses an SVM model, a scaler, and a list of features to compute ranking scores."""
    # 1. 'feature_names' now contains the raw feature names (e.g., "component_similarity")
    feature_cols = feature_names

    if not all(col in documents_df.columns for col in feature_cols):
        print("ERROR: Document file is missing required feature columns.")
        return None

    # 2. Extract ONLY the numeric features
    doc_features = documents_df[feature_cols]

    # 3. APPLY THE SCALER (use .transform())
    # This works because the scaler was trained on raw features
    doc_features_scaled = scaler.transform(doc_features)


    # 4. Extract the weight vector 'w'
    w = model.coef_[0]

    # 5. Compute scores ON THE SCALED DATA
    ranking_scores = np.dot(doc_features_scaled, w)

    # Add scores and sort
    ranked_df = documents_df.copy()
    ranked_df['ranking_score'] = ranking_scores
    ranked_df = ranked_df.sort_values(by='ranking_score', ascending=False).reset_index(drop=True)

    return ranked_df


def main():
    """
    Loads a model, a scaler, and feature names and uses them to rank documents.
    """
    # 1. Load model, scaler, and feature names
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(FEATURE_NAMES_FILE, 'r') as f:
            feature_names = json.load(f)

    except FileNotFoundError as e:
        print(f"ERROR: Missing file. Could not load model, scaler, or feature names. {e}")
        return

    # 2. Load the documents to be ranked
    try:
        docs_to_rank_df = pd.read_csv(EXAMPLE_FEATURE_FILE)
    except FileNotFoundError:
        print(f"ERROR: Document feature file '{EXAMPLE_FEATURE_FILE}' not found.")
        return

    # 3. Get the new ranking
    final_ranking = rank_documents(model, scaler, docs_to_rank_df, feature_names)

    if final_ranking is not None:
        print("\nTop 5 ranked documents based on the learned model:")
        print(final_ranking[['loinc_num', 'long_common_name', 'ranking_score']].head(10))


if __name__ == "__main__":
    main()
