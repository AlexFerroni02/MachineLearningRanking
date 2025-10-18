import config  # Imports all paths and parameters from config.py
import os
import shutil # For optionally clearing directories

# Import the main functions from your pipeline modules
from ml_pipeline import data_processing, simulation, training_prep, training

# --- Optional: Function to clear previous intermediate data ---
def clear_intermediate_data():
    """Removes generated data folders for a clean run."""
    print("Clearing intermediate data directories...")
    dirs_to_remove = [
        config.FEATURE_RANKINGS_DIR,
        config.CLICK_LOGS_DIR,
    ]
    files_to_remove = [
        config.CONCEPTUAL_PAIRS_FILE,
    ]

    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"  Removed directory: {dir_path}")
            except OSError as e:
                print(f"  Error removing directory {dir_path}: {e}")

    for file_path in files_to_remove:
         if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  Removed file: {file_path}")
            except OSError as e:
                print(f"  Error removing file {file_path}: {e}")

# --- Main Pipeline Execution ---
def main():
    """
    Orchestrates the entire LTR model training pipeline,
    with a pause after click simulation for manual review.
    """
    print("--- STARTING LTR MODEL TRAINING PIPELINE ---")

    # --- Optional: Clean previous run ---
    CLEAN_RUN = True
    if CLEAN_RUN:
        clear_intermediate_data()
        os.makedirs(config.MODELS_DIR, exist_ok=True) # Ensure models dir exists
    # ---

    # --- Phase 1: Feature Engineering ---
    print("\n[PHASE 1/3] Running Feature Engineering...")
    success = data_processing.run_feature_engineering(
        xls_file_path=config.XLS_FILE,
        output_dir=config.FEATURE_RANKINGS_DIR,
        system_map=config.SYSTEM_MAP,
        property_map=config.PROPERTY_MAP,
        loinc_text_columns=config.LOINC_TEXT_COLUMNS
    )
    if not success:
         print("Feature engineering failed. Aborting pipeline.")
         return
    print("[PHASE 1/3] Completed.")

    # --- Phase 2: Click Simulation ---
    print("\n[PHASE 2/3] Running Click Simulation...")
    success = simulation.run_click_simulation(
        input_dir=config.FEATURE_RANKINGS_DIR,
        output_dir=config.CLICK_LOGS_DIR,
        prob_param=config.CLICK_PROBABILITY_PARAM # Using the simple model param
    )
    if not success:
         print("Click simulation failed. Aborting pipeline.")
         return
    print("[PHASE 2/3] Completed.")

    # --- PAUSE FOR MANUAL CHECK ---
    print("\n-----------------------------------------------------")
    print(f">>> PIPELINE PAUSED <<<")
    print(f"Please manually inspect the generated click log files in:")
    print(f"    {os.path.abspath(config.CLICK_LOGS_DIR)}")
    print("Check the 'clicked' column distribution.")
    input("Press Enter in this terminal to continue the pipeline...")
    print("-----------------------------------------------------")
    # --- END PAUSE ---


    # --- Phase 3: Training Data Preparation (Pair Creation + Transformation) ---
    print("\n[PHASE 3/4] Running Training Data Preparation...")
    # This phase now reads from the click logs you just checked
    success, feature_names_used = training_prep.run_training_preparation(
        click_logs_dir=config.CLICK_LOGS_DIR,
        conceptual_pairs_file=config.CONCEPTUAL_PAIRS_FILE,
        feature_rankings_dir=config.FEATURE_RANKINGS_DIR,
        svm_training_dataset_file=config.SVM_TRAINING_DATASET_FILE # Path from config
    )
    if not success:
         print("Training data preparation failed. Aborting.")
         return
    print("[PHASE 3/4] Completed.")


    # --- Phase 4: Model Training ---
    print("\n[PHASE 4/4] Running Model Training...")
    success_train = training.train_model(
         conceptual_pairs_file=config.CONCEPTUAL_PAIRS_FILE, # Still needed for doc IDs
         feature_rankings_dir=config.FEATURE_RANKINGS_DIR,  # Needed for scaler fitting
         model_output_file=config.MODEL_OUTPUT_FILE,
         scaler_output_file=config.SCALER_OUTPUT_FILE,
         feature_names_output_file=config.FEATURE_NAMES_FILE,
         grid_search_params=config.GRID_SEARCH_PARAMS
    )
    if not success_train:
        print("Model training failed.")
        return
    print("[PHASE 4/4] Completed.")


    print("\n--- LTR MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY ---")
    # ... (Final print messages) ...

if __name__ == "__main__":
    # Ensure necessary parameters are in config.py
    required_configs = ['XLS_FILE', 'FEATURE_RANKINGS_DIR', 'CLICK_LOGS_DIR',
                        'CONCEPTUAL_PAIRS_FILE', 'MODELS_DIR', 'MODEL_OUTPUT_FILE',
                        'SCALER_OUTPUT_FILE', 'FEATURE_NAMES_FILE', 'SYSTEM_MAP',
                        'PROPERTY_MAP', 'LOINC_TEXT_COLUMNS', 'CLICK_PROBABILITY_PARAM',
                        'GRID_SEARCH_PARAMS', 'SVM_TRAINING_DATASET_FILE'] # Added SVM training file path
    missing_configs = [cfg for cfg in required_configs if not hasattr(config, cfg)]
    if missing_configs:
        print("ERROR: Your config.py file is missing the following required variables:")
        for cfg in missing_configs:
            print(f"- {cfg}")
    else:
        main()