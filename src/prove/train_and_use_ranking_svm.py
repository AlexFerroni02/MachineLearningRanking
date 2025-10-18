import os

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Used for saving and loading the trained model
import numpy as np

# --- 1. CONFIGURATION ---
# The final training dataset we created in the last step
SVM_TRAINING_DATASET_FILE = "../../data/svm_training_dataset.csv"

# Where to save the final, trained model
MODEL_OUTPUT_FILE = "../ranking_svm_model.joblib"

# The file containing the features for all documents (needed for the ranking demonstration)
# We just pick one of the query files as an example data source
EXAMPLE_FEATURE_FILE = "../../data/feature_rankings/glucose_in_blood.csv"


# --- 2. MODEL TRAINING FUNCTION ---

def train_svm_model():
    """
    Loads the training data, trains the Ranking SVM model, and saves it to a file.
    """
    print("--- Starting Model Training ---")

    # 1. Load the training data
    try:
        df_train = pd.read_csv(SVM_TRAINING_DATASET_FILE)
    except FileNotFoundError:
        print(f"ERROR: Training file '{SVM_TRAINING_DATASET_FILE}' not found. Please run the previous script.")
        return

    print(f"Loaded {len(df_train)} training examples.")

    # 2. Separate features (X) from labels (y)
    X = df_train.drop('label', axis=1)
    y = df_train['label']

    # 3. Split data for training and testing (optional but good practice)
    # This helps us evaluate how well the model learned.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} examples, testing on {len(X_test)} examples.")

    # 4. Create and train the SVM classifier
    # We use a linear kernel as is standard for Ranking SVM.
    # The C parameter controls the trade-off between a smooth decision boundary and classifying
    # training points correctly.
    print("Training the SVM classifier... (this may take a moment)")
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate the model on the test set
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on the test set: {accuracy:.2f}")

    # 6. Save the trained model to a file
    joblib.dump(svm_classifier, MODEL_OUTPUT_FILE)
    print(f"Model saved successfully to '{MODEL_OUTPUT_FILE}'")

    return svm_classifier


# --- 3. RANKING DEMONSTRATION FUNCTION ---

def rank_documents_with_model(model, documents_df):
    """
    Uses the trained SVM model to calculate ranking scores for a new set of documents.
    """
    print("\n--- Demonstrating Ranking with the Trained Model ---")

    # 1. Identify the feature columns from the document DataFrame
    # This must match the features the model was trained on (without the '_diff' suffix)
    feature_cols = [col.replace('_diff', '') for col in model.feature_names_in_]
    doc_features = documents_df[feature_cols]

    # 2. Extract the learned weight vector 'w' from the linear SVM model
    # For a linear kernel, this is stored in the .coef_ attribute
    w = model.coef_[0]

    # 3. Calculate the ranking score for each document
    # Score = w · document_feature_vector
    ranking_scores = np.dot(doc_features.values, w)

    # 4. Add the scores to the DataFrame and sort to get the final ranking
    ranked_df = documents_df.copy()
    ranked_df['ranking_score'] = ranking_scores
    ranked_df = ranked_df.sort_values(by='ranking_score', ascending=False).reset_index(drop=True)

    print("Top 5 ranked documents based on the learned model:")
    print(ranked_df[['loinc_num', 'long_common_name', 'ranking_score']].head(5))

    return ranked_df


# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    # First, train the model (or load it if it already exists)
    if not os.path.exists(MODEL_OUTPUT_FILE):
        model = train_svm_model()
    else:
        print(f"Loading existing model from '{MODEL_OUTPUT_FILE}'...")
        model = joblib.load(MODEL_OUTPUT_FILE)

    if model:
        # Now, demonstrate how to use the model for ranking
        try:
            # Load some sample documents to rank
            sample_docs_df = pd.read_csv(EXAMPLE_FEATURE_FILE)
            rank_documents_with_model(model, sample_docs_df)
        except FileNotFoundError:
            print(f"\nERROR: Could not run ranking demonstration because the example feature file was not found.")
            print(f"Please ensure '{EXAMPLE_FEATURE_FILE}' exists.")