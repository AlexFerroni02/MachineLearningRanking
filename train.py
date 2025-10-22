import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import glob

# --- 1. CONFIGURATION ---
CONCEPTUAL_PAIRS_FILE = "data/conceptual_preference_pairs.csv"
FEATURE_RANKINGS_DIR = "data/feature_rankings"
MODEL_OUTPUT_FILE = "model/ranking_svm_model_optimized.joblib"
SCALER_OUTPUT_FILE = "model/feature_scaler.joblib"
FEATURE_NAMES_FILE = "model/feature_names.json"

# --- 2. SUPPORT FUNCTIONS ---
def get_feature_columns(df):
    """Identifies numeric feature columns (excluding '_diff')."""
    exclude_cols = ['loinc_num', 'long_common_name', 'component', 'system', 'property', 'clicked']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    return feature_cols

def load_all_features(feature_dir):
    """Loads and merges all raw features from all ranking files."""
    all_dfs = []
    for file_path in glob.glob(os.path.join(feature_dir, "*.csv")):
        all_dfs.append(pd.read_csv(file_path))

    # Merge all documents into a single DataFrame and remove duplicates
    df_all_features = pd.concat(all_dfs).drop_duplicates(subset=['loinc_num']).set_index('loinc_num')
    return df_all_features

def create_difference_vectors(pairs_df, features_df, feature_cols, scaler):
    """
    Creates scaled difference vectors from preference pairs
    and a PRE-TRAINED scaler.
    """
    vectors = []
    labels = []

    for index, row in pairs_df.iterrows():
        try:
            vec_pref = features_df.loc[row['preferred_doc_id'], feature_cols]
            vec_not_pref = features_df.loc[row['not_preferred_doc_id'], feature_cols]

            # Transform raw vectors using the scaler
            vec_pref_scaled = scaler.transform(vec_pref.values.reshape(1, -1))[0]
            vec_not_pref_scaled = scaler.transform(vec_not_pref.values.reshape(1, -1))[0]

            # Compute the difference on scaled data
            vectors.append(vec_pref_scaled - vec_not_pref_scaled)
            labels.append(1)

            # Add the inverse pair
            vectors.append(vec_not_pref_scaled - vec_pref_scaled)
            labels.append(-1)

        except KeyError as e:
            # Skip the pair if an ID is not found
            continue

    return pd.DataFrame(vectors, columns=[f"{col}_diff" for col in feature_cols]), pd.Series(labels, name="label")
# --- 3. MAIN FUNCTION ---

def main():
    # 1. Load conceptual preference pairs
    try:
        df_pairs = pd.read_csv(CONCEPTUAL_PAIRS_FILE)
    except FileNotFoundError:
        print(f"ERROR: File '{CONCEPTUAL_PAIRS_FILE}' not found.")
        return

    # 2. Load ALL raw features for ALL documents
    df_all_features = load_all_features(FEATURE_RANKINGS_DIR)
    feature_columns = get_feature_columns(df_all_features)

    # 3. Split preference pairs into Train and Test
    # This is the FUNDAMENTAL step to avoid data leakage
    pairs_train, pairs_test = train_test_split(df_pairs, test_size=0.2, random_state=42)

    # 4. Train the Scaler ONLY on features of documents appearing in the TRAIN set
    train_doc_ids = pd.unique(np.concatenate((pairs_train['preferred_doc_id'], pairs_train['not_preferred_doc_id'])))
    train_features = df_all_features.loc[train_doc_ids, feature_columns]

    scaler = StandardScaler()
    scaler.fit(train_features)

    # 5. Create SCALED difference vectors for training and test
    X_train, y_train = create_difference_vectors(pairs_train, df_all_features, feature_columns, scaler)

    X_test, y_test = create_difference_vectors(pairs_test, df_all_features, feature_columns, scaler)

    # 6. Run GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100]}
    svm = SVC(kernel='linear')
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)


    grid_search.fit(X_train, y_train)

    # 7. Get and evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal model accuracy on the test set: {accuracy:.2f}")

    # 8. Save model, scaler, and feature names
    joblib.dump(best_model, MODEL_OUTPUT_FILE)
    joblib.dump(scaler, SCALER_OUTPUT_FILE)
    # Save RAW feature names (without _diff)
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(feature_columns, f)

    # 9. Analyze model weights
    print("\n--- Model Weights (Feature Importance) ---")
    weights = best_model.coef_[0]
    df_weights = pd.DataFrame({'feature': X_train.columns, 'weight': weights})
    print(df_weights.sort_values(by='weight', ascending=False))

if __name__ == "__main__":
    main()
