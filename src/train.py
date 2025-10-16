import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import os

# --- 1. CONFIGURATION ---
SVM_TRAINING_DATASET_FILE = "../data/svm_training_dataset.csv"
MODEL_OUTPUT_FILE = "ranking_svm_model_optimized.joblib"


# --- 2. MAIN FUNCTION ---

def main():
    """
    Loads the training data, finds the best hyperparameters using GridSearchCV,
    trains the final model, and saves it.
    """
    print("--- Starting Optimized Model Training ---")

    # 1. Load the training data
    try:
        df_train = pd.read_csv(SVM_TRAINING_DATASET_FILE)
    except FileNotFoundError:
        print(f"ERROR: Training file '{SVM_TRAINING_DATASET_FILE}' not found.")
        return

    print(f"Loaded {len(df_train)} training examples.")

    X = df_train.drop('label', axis=1)
    y = df_train['label']

    # 2. Split data for training and final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {len(X_train)} for training/cross-validation, {len(X_test)} for final testing.")

    # 3. Define the hyperparameter grid to search
    # This dictionary tells GridSearchCV which parameters to test and which values to try.
    # We are testing a range of values for the regularization parameter 'C'.
    param_grid = {
        'C': [0.1, 1, 10, 100]
    }
    print(f"Searching for the best 'C' value in: {param_grid['C']}")

    # 4. Set up and run GridSearchCV
    # GridSearchCV will automatically use Cross-Validation (cv=5 means 5-fold CV)
    # to find the best 'C' value from the param_grid.
    # verbose=2 provides detailed logs during the process.
    svm = SVC(kernel='linear')
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

    print("Starting GridSearchCV... (this will take longer than the simple training)")
    grid_search.fit(X_train, y_train)

    # 5. Get the best model and its parameters
    print("\nGridSearchCV complete.")
    print(f"The best C value found is: {grid_search.best_params_['C']}")
    best_model = grid_search.best_estimator_

    # 6. Evaluate the best model on the held-out test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal model accuracy on the test set: {accuracy:.2f}")

    # 7. Save the final, optimized model
    joblib.dump(best_model, MODEL_OUTPUT_FILE)
    print(f"Optimized model saved successfully to '{MODEL_OUTPUT_FILE}'")


if __name__ == "__main__":
    main()