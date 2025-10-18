import pandas as pd
import numpy as np
import os
import glob
import json

# --- Helper Functions (Internal) ---

def _get_feature_columns(df):
    """Identifies numeric feature columns, excluding IDs and text."""
    exclude_cols = ['loinc_num', 'long_common_name', 'component', 'system', 'property', 'clicked']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    return feature_cols

def _create_preference_pairs(input_dir, output_file):
    """Generates the conceptual preference pairs file from click logs."""
    click_log_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not click_log_files:
        print(f"WARNING: No click log files found in '{input_dir}'.")
        return False, None

    all_preference_pairs = []
    total_pairs_generated = 0

    for file_path in click_log_files:
        filename = os.path.basename(file_path)
        query_text = filename.replace('.csv', '').replace('_', ' ')

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}. Skipping.")
            continue

        clicked_indices = df[df['clicked'] == True].index
        if len(clicked_indices) == 0:
            continue # Skip files with no clicks silently

        for i in clicked_indices:
            preferred_doc_id = df.loc[i, 'loinc_num']
            for j in range(i):
                if not df.loc[j, 'clicked']:
                    not_preferred_doc_id = df.loc[j, 'loinc_num']
                    pair_data = {
                        'query': query_text,
                        'preferred_doc_id': preferred_doc_id,
                        'not_preferred_doc_id': not_preferred_doc_id,
                        'label': 1
                    }
                    all_preference_pairs.append(pair_data)
                    total_pairs_generated += 1

    if not all_preference_pairs:
        print("WARNING: No preference pairs were generated from any files.")
        return False, None

    df_pairs = pd.DataFrame(all_preference_pairs)
    df_pairs.to_csv(output_file, index=False)
    print(f"Conceptual pairs saved ({total_pairs_generated} pairs).") # Concise status
    return True, df_pairs


def _transform_pairs_for_svm(df_pairs, feature_rankings_dir, output_file):
    """Transforms conceptual pairs into difference vectors for SVM training."""
    if df_pairs is None or df_pairs.empty:
        print("ERROR: No conceptual pairs provided for SVM transformation.")
        return False, None

    all_training_vectors = []
    feature_columns = None
    total_svm_examples = 0

    for query_text, group in df_pairs.groupby('query'):
        safe_filename = query_text.lower().replace(" ", "_") + ".csv"
        feature_file_path = os.path.join(feature_rankings_dir, safe_filename)

        try:
            df_features = pd.read_csv(feature_file_path)
            df_features.set_index('loinc_num', inplace=True)
            if feature_columns is None:
                feature_columns = _get_feature_columns(df_features)
                if not feature_columns:
                    print(f"FATAL ERROR: No numeric features found in {feature_file_path}. Aborting.")
                    return False, None
                print(f"Identified {len(feature_columns)} feature columns.") # Keep this important info

        except FileNotFoundError:
            print(f"  WARNING: Feature file not found for query '{query_text}'. Skipping pairs.")
            continue
        except Exception as e:
            print(f"  ERROR loading feature file {feature_file_path}: {e}. Skipping pairs.")
            continue

        for index, row in group.iterrows():
            pref_id = row['preferred_doc_id']
            not_pref_id = row['not_preferred_doc_id']

            try:
                vec_pref = df_features.loc[pref_id, feature_columns]
                vec_not_pref = df_features.loc[not_pref_id, feature_columns]

                diff_vector = vec_pref.values - vec_not_pref.values
                all_training_vectors.append(np.append(diff_vector, 1))

                inverse_diff = vec_not_pref.values - vec_pref.values
                all_training_vectors.append(np.append(inverse_diff, -1))
                total_svm_examples += 2

            except KeyError as e:
                # Skip silently or add a very minimal warning if needed often
                # print(f"  Skipping pair due to missing ID: {e}")
                continue
            except Exception as e:
                print(f"  ERROR processing pair ({pref_id}, {not_pref_id}): {e}. Skipping.")
                continue

    if not all_training_vectors:
        print("ERROR: No SVM training vectors were generated.")
        return False, None
    if feature_columns is None:
         print("ERROR: Could not determine feature columns.")
         return False, None


    feature_diff_columns = [f"{col}_diff" for col in feature_columns]
    final_columns = feature_diff_columns + ['label']

    df_training = pd.DataFrame(all_training_vectors, columns=final_columns)
    df_training.to_csv(output_file, index=False)
    print(f"SVM training dataset saved ({total_svm_examples} examples).") # Concise status
    return True, feature_columns


# --- Main Function (to be imported by train.py) ---

def run_training_preparation(click_logs_dir, conceptual_pairs_file, feature_rankings_dir, svm_training_dataset_file):
    """
    Orchestrates the two steps of training data preparation:
    1. Create conceptual preference pairs.
    2. Transform pairs into SVM training data (difference vectors).
    """
    print(f"\n--- Starting Training Data Preparation ---") # Main status

    # Step 1: Create Conceptual Pairs
    success_pairs, df_pairs = _create_preference_pairs(click_logs_dir, conceptual_pairs_file)
    if not success_pairs:
        return False, None

    # Step 2: Transform for SVM
    success_svm, feature_names = _transform_pairs_for_svm(df_pairs, feature_rankings_dir, svm_training_dataset_file)

    if success_svm:
        print(f"--- Training Data Preparation Complete ---") # Final status
    else:
        print(f"--- Training Data Preparation Failed ---")

    return success_svm, feature_names