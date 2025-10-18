# ranker.py

import pandas as pd
import joblib
import numpy as np
import os
import json
import config # Import paths and settings from config.py

def rank_documents(model, scaler, documents_df, feature_names):
    """
    Uses a trained SVM model, scaler, and feature list
    to calculate ranking scores for a set of documents.
    """
    print("\n--- Ranking Documents with Trained Model and Scaler ---")

    # 1. 'feature_names' now contains the raw feature names (e.g., "component_similarity")
    feature_cols = feature_names

    # Check if all required feature columns are present in the input DataFrame
    missing_cols = [col for col in feature_cols if col not in documents_df.columns]
    if missing_cols:
        print(f"ERROR: Document file is missing required feature columns: {missing_cols}")
        return None

    # 2. Extract ONLY the numeric feature columns in the correct order
    try:
        doc_features = documents_df[feature_cols]
    except KeyError as e:
         print(f"ERROR: Could not select feature columns. Missing: {e}")
         return None
    except Exception as e:
         print(f"ERROR during feature selection: {e}")
         return None


    # 3. APPLY THE SCALER (use .transform())
    # This now works because the scaler was trained on raw features
    try:
        doc_features_scaled = scaler.transform(doc_features)
        print("Features successfully scaled using the loaded scaler.")
    except ValueError as e:
        print(f"ERROR applying scaler: {e}")
        print("Ensure the number and order of features in the input match the training.")
        return None
    except Exception as e:
        print(f"ERROR during scaling: {e}")
        return None


    # 4. Extract the weight vector 'w'
    w = model.coef_[0]

    # 5. Calculate scores ON SCALED DATA
    ranking_scores = np.dot(doc_features_scaled, w)

    # Add scores and sort
    ranked_df = documents_df.copy()
    ranked_df['ranking_score'] = ranking_scores
    ranked_df = ranked_df.sort_values(by='ranking_score', ascending=False).reset_index(drop=True)

    return ranked_df


def main():
    """
    Loads a pre-trained model, scaler, and feature names and uses them to rank documents.
    """
    print("--- LTR Document Ranker ---")

    # Determine a default example file if needed, or allow user input/argument parsing later
    # For now, using the path from config.py
    example_file_path = os.path.join(config.FEATURE_RANKINGS_DIR, "glucose_in_blood.csv") # Example

    # 1. Load model, scaler, and feature names using paths from config.py
    try:
        model = joblib.load(config.MODEL_OUTPUT_FILE)
        scaler = joblib.load(config.SCALER_OUTPUT_FILE)
        with open(config.FEATURE_NAMES_FILE, 'r') as f:
            feature_names = json.load(f)
        print(f"Successfully loaded model from: {config.MODEL_OUTPUT_FILE}")
        print(f"Successfully loaded scaler from: {config.SCALER_OUTPUT_FILE}")
        print(f"Successfully loaded feature names from: {config.FEATURE_NAMES_FILE}")
    except FileNotFoundError as e:
        print(f"ERROR: Missing required file. Could not load model, scaler, or feature names.")
        print(f"  File not found: {e.filename}")
        print("Please run the training pipeline first (`python train.py`).")
        return
    except Exception as e:
        print(f"ERROR loading files: {e}")
        return


    # 2. Load the documents to rank (using the example file path)
    try:
        docs_to_rank_df = pd.read_csv(example_file_path)
        print(f"\nLoaded {len(docs_to_rank_df)} documents to rank from: {example_file_path}")
    except FileNotFoundError:
        print(f"ERROR: Example document feature file '{example_file_path}' not found.")
        print("Ensure the feature engineering step has been run and the file exists.")
        return
    except Exception as e:
        print(f"ERROR loading document file: {e}")
        return

    # 3. Get the new ranking
    final_ranking = rank_documents(model, scaler, docs_to_rank_df, feature_names)

    if final_ranking is not None:
        print("\nTop 5 ranked documents based on the learned model:")
        # Display relevant columns
        # Display 'long_common_name' and 'ranking_score'
        print(final_ranking[['long_common_name', 'ranking_score']].head(5))


if __name__ == "__main__":
    main()