import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import glob # Keep import for load_all_features if moved here later

# --- Helper Functions (Could potentially be moved to a utils module) ---

def get_feature_columns_from_df(df):
    """Identifies numeric feature columns from a DataFrame."""
    exclude_cols = ['loinc_num', 'long_common_name', 'component', 'system', 'property', 'clicked']
    # Filter based on exclusion list and numeric type
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    return feature_cols

def load_all_features(feature_dir, feature_columns):
    """
    Loads raw features for specified documents, ensuring correct columns.
    Used for fitting the scaler.
    """
    all_dfs = []
    required_cols = ['loinc_num'] + feature_columns # Ensure loinc_num is loaded
    try:
        for file_path in glob.glob(os.path.join(feature_dir, "*.csv")):
            # Load only necessary columns for efficiency
            df_temp = pd.read_csv(file_path, usecols=lambda c: c in required_cols)
            all_dfs.append(df_temp)

        if not all_dfs:
            print(f"ERROR: No feature files found or loaded from '{feature_dir}'.")
            return None

        # Concatenate, drop duplicates based on loinc_num, and set index
        df_all_features = pd.concat(all_dfs).drop_duplicates(subset=['loinc_num']).set_index('loinc_num')
        # Re-select columns to ensure correct order and presence
        df_all_features = df_all_features[feature_columns]
        return df_all_features
    except Exception as e:
        print(f"ERROR loading feature files: {e}")
        return None


def create_difference_vectors(pairs_df, features_df, feature_cols, scaler):
    """
    Creates scaled difference vectors from preference pairs and a fitted scaler.
    """
    vectors = []
    labels = []
    skipped_count = 0

    # Ensure features_df has loinc_num as index if not already
    if features_df.index.name != 'loinc_num':
         if 'loinc_num' in features_df.columns:
             features_df = features_df.set_index('loinc_num')
         else:
             print("ERROR: features_df must have 'loinc_num' as index or column.")
             return None, None # Indicate error

    for index, row in pairs_df.iterrows():
        try:
            # Check if IDs exist before trying to access .loc
            pref_id = row['preferred_doc_id']
            not_pref_id = row['not_preferred_doc_id']
            if pref_id not in features_df.index or not_pref_id not in features_df.index:
                skipped_count += 1
                continue # Skip if either ID is missing

            vec_pref = features_df.loc[pref_id, feature_cols]
            vec_not_pref = features_df.loc[not_pref_id, feature_cols]

            # Scale raw feature vectors using the pre-fitted scaler
            vec_pref_scaled = scaler.transform(vec_pref.values.reshape(1, -1))[0]
            vec_not_pref_scaled = scaler.transform(vec_not_pref.values.reshape(1, -1))[0]

            # Calculate difference on scaled data
            vectors.append(vec_pref_scaled - vec_not_pref_scaled)
            labels.append(1)

            # Add inverse pair
            vectors.append(vec_not_pref_scaled - vec_pref_scaled)
            labels.append(-1)

        except KeyError as e:
             skipped_count += 1 # Should theoretically not happen due to check above
             continue
        except Exception as e:
             print(f"  ERROR processing pair ({pref_id}, {not_pref_id}): {e}. Skipping.")
             skipped_count += 1
             continue

    if skipped_count > 0:
        print(f"  WARNING: Skipped {skipped_count} pairs due to missing document IDs in feature files.")

    if not vectors:
        return None, None # Indicate error if no vectors were created

    # Return DataFrame and Series
    diff_cols = [f"{col}_diff" for col in feature_cols]
    return pd.DataFrame(vectors, columns=diff_cols), pd.Series(labels, name="label")


# --- Main Training Function (to be imported by train.py) ---

def train_model(conceptual_pairs_file, feature_rankings_dir, model_output_file, scaler_output_file, feature_names_output_file, grid_search_params):
    """
    Loads conceptual pairs, prepares data (including scaling),
    runs GridSearchCV, trains the final SVM model, and saves outputs.

    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"\n--- Starting Model Training Phase ---")

    # 1. Load conceptual preference pairs
    try:
        df_pairs = pd.read_csv(conceptual_pairs_file)
        if df_pairs.empty:
            print("ERROR: Conceptual pairs file is empty.")
            return False
    except FileNotFoundError:
        print(f"ERROR: Conceptual pairs file '{conceptual_pairs_file}' not found.")
        return False
    print(f"Loaded {len(df_pairs)} conceptual pairs.")

    # 2. Determine base feature columns from the first feature file
    try:
        first_feature_file = glob.glob(os.path.join(feature_rankings_dir, "*.csv"))[0]
        df_temp = pd.read_csv(first_feature_file)
        base_feature_names = get_feature_columns_from_df(df_temp)
        if not base_feature_names:
            print("ERROR: Could not determine feature columns from feature files.")
            return False
        print(f"Identified base feature columns: {base_feature_names}")
    except IndexError:
        print(f"ERROR: No feature files (*.csv) found in '{feature_rankings_dir}'.")
        return False
    except Exception as e:
        print(f"ERROR reading feature file to determine columns: {e}")
        return False


    # 3. Load ALL raw features for fitting the scaler
    df_all_features = load_all_features(feature_rankings_dir, base_feature_names)
    if df_all_features is None:
        print("ERROR: Failed to load raw features for scaler fitting.")
        return False
    print(f"Loaded raw features for {len(df_all_features)} unique documents.")


    # 4. Split conceptual pairs into Train and Test sets
    pairs_train, pairs_test = train_test_split(df_pairs, test_size=0.2, random_state=42)
    print(f"Conceptual pairs split: {len(pairs_train)} train, {len(pairs_test)} test.")

    # 5. Fit the Scaler ONLY on features of documents present in the training pairs
    train_doc_ids = pd.unique(np.concatenate((pairs_train['preferred_doc_id'], pairs_train['not_preferred_doc_id'])))

    # Ensure IDs exist in the loaded features before fitting scaler
    train_doc_ids_present = df_all_features.index.intersection(train_doc_ids)
    if len(train_doc_ids_present) == 0:
        print("ERROR: None of the document IDs from training pairs were found in the feature files.")
        return False
    if len(train_doc_ids_present) < len(train_doc_ids):
         print(f"WARNING: {len(train_doc_ids) - len(train_doc_ids_present)} document IDs from training pairs not found in feature files.")

    train_features_for_scaler = df_all_features.loc[train_doc_ids_present, base_feature_names]

    print("Fitting StandardScaler ONLY on training document features...")
    scaler = StandardScaler()
    scaler.fit(train_features_for_scaler)
    print("Scaler fitted.")

    # 6. Create scaled difference vectors for train and test sets
    print("Creating scaled difference vectors for training set...")
    X_train, y_train = create_difference_vectors(pairs_train, df_all_features, base_feature_names, scaler)
    if X_train is None: return False # Check for errors

    print("Creating scaled difference vectors for test set...")
    X_test, y_test = create_difference_vectors(pairs_test, df_all_features, base_feature_names, scaler)
    if X_test is None: return False # Check for errors

    if X_train.empty or X_test.empty:
        print("ERROR: Training or test set difference vectors are empty. Cannot proceed.")
        return False
    print(f"Created {len(X_train)} training examples and {len(X_test)} test examples.")

    # 7. Perform GridSearchCV
    print(f"Performing GridSearchCV with params: {grid_search_params}")
    svm = SVC(kernel='linear', probability=False) # Probability=False speeds up training if not needed
    grid_search = GridSearchCV(estimator=svm, param_grid=grid_search_params, cv=5, verbose=1, n_jobs=-1) # Reduced verbosity

    grid_search.fit(X_train, y_train)

    # 8. Get best model and evaluate
    print("\nGridSearchCV complete.")
    print(f"Best C value found: {grid_search.best_params_['C']}")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final model accuracy on test set: {accuracy:.4f}")

    # 9. Save model, scaler, and feature names
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)

    joblib.dump(best_model, model_output_file)
    joblib.dump(scaler, scaler_output_file)
    # Save the BASE feature names (without _diff)
    with open(feature_names_output_file, 'w') as f:
        json.dump(base_feature_names, f)

    print(f"Model saved to '{model_output_file}'")
    print(f"Scaler saved to '{scaler_output_file}'")
    print(f"Feature names saved to '{feature_names_output_file}'")

    # 10. Analyze weights (optional, can be moved to a separate analysis script)
    try:
        print("\n--- Model Feature Weights (Importance) ---")
        weights = best_model.coef_[0]
        # Use the derived diff column names for display
        df_weights = pd.DataFrame({'feature': X_train.columns, 'weight': weights})
        print(df_weights.sort_values(by='weight', ascending=False))
    except Exception as e:
        print(f"Could not display feature weights: {e}")


    print(f"\n--- Model Training Phase Completed! ---")
    return True # Indicate success