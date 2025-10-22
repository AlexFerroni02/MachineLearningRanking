import pandas as pd
import numpy as np
import os
# --- 1. CONFIGURATION ---
# The file with the preference pairs created by the previous script
CONCEPTUAL_PAIRS_FILE = "../data/conceptual_preference_pairs.csv"

# The folder where the CSV files with features for each query are located
FEATURE_RANKINGS_DIR = "../data/feature_rankings"

# The final output file, ready to be used for SVM training
SVM_TRAINING_DATASET_FILE = "../data/svm_training_dataset.csv"

# --- 2. SUPPORT FUNCTION ---

def get_feature_columns(df):
    """Automatically identifies all numeric feature columns."""
    exclude_cols = ['loinc_num', 'long_common_name', 'component', 'system', 'property', 'clicked']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    return feature_cols

# --- 3. MAIN FUNCTION ---

def main():
    """
    Orchestrates the final dataset transformation:
    1. Loads the conceptual preference pairs.
    2. Groups pairs by query for efficiency.
    3. For each query, loads the features and computes difference vectors.
    4. Saves the final training dataset.
    """
    try:
        df_pairs = pd.read_csv(CONCEPTUAL_PAIRS_FILE)
    except FileNotFoundError:
        print(f"ERROR: Conceptual pairs file '{CONCEPTUAL_PAIRS_FILE}' not found.")
        return

    all_training_vectors = []

    # Group by query. This allows us to load each feature file only once.
    for query_text, group in df_pairs.groupby('query'):

        # Build the path to the corresponding feature file
        safe_filename = query_text.lower().replace(" ", "_") + ".csv"
        feature_file_path = os.path.join(FEATURE_RANKINGS_DIR, safe_filename)

        try:
            df_features = pd.read_csv(feature_file_path)
            # Use loinc_num as index for fast lookup
            df_features.set_index('loinc_num', inplace=True)
        except FileNotFoundError:
            print(f"  -> WARNING: Feature file '{feature_file_path}' not found. Skipping this query.")
            continue

        feature_columns = get_feature_columns(df_features)
        if not feature_columns:
            print("  -> WARNING: No feature columns found in the file. Skipping this query.")
            continue

        # Iterate over each preference pair for this query
        for index, row in group.iterrows():
            pref_id = row['preferred_doc_id']
            not_pref_id = row['not_preferred_doc_id']
            try:
                # Extract the numeric feature vectors for both documents
                vec_pref = df_features.loc[pref_id, feature_columns]
                vec_not_pref = df_features.loc[not_pref_id, feature_columns]

                # Compute the difference vector
                diff_vector = vec_pref.values - vec_not_pref.values
                all_training_vectors.append(np.append(diff_vector, 1))  # Add label +1

                # Also add the inverse pair for more robust training
                inverse_diff = vec_not_pref.values - vec_pref.values
                all_training_vectors.append(np.append(inverse_diff, -1))  # Add label -1

            except KeyError as e:
                print(f"  -> WARNING: Document ID {e} not found in the feature file. Skipping this pair.")

    if not all_training_vectors:
        print("\n--- No training vectors were generated. ---")
        return

    # Create the final DataFrame ready for SVM
    feature_diff_columns = [f"{col}_diff" for col in feature_columns]
    final_columns = feature_diff_columns + ['label']

    df_training = pd.DataFrame(all_training_vectors, columns=final_columns)

    df_training.to_csv(SVM_TRAINING_DATASET_FILE, index=False)

if __name__ == "__main__":
    main()
