import os

# --- Base Paths ---
DATA_DIR = "data"
MODELS_DIR = "models"

# --- Raw Data Paths ---
XLS_FILE = os.path.join(DATA_DIR, "loinc_dataset-v2.xlsx")

# --- Intermediate Data Paths ---
FEATURE_RANKINGS_DIR = os.path.join(DATA_DIR, "feature_rankings")
CLICK_LOGS_DIR = os.path.join(DATA_DIR, "click_logs")
CONCEPTUAL_PAIRS_FILE = os.path.join(DATA_DIR, "conceptual_preference_pairs.csv")
SVM_TRAINING_DATASET_FILE = os.path.join(DATA_DIR, "svm_training_dataset.csv") # This will be deleted

# --- Final Model Paths ---
MODEL_OUTPUT_FILE = os.path.join(MODELS_DIR, "ranking_svm_model_optimized.joblib")
SCALER_OUTPUT_FILE = os.path.join(MODELS_DIR, "feature_scaler.joblib")
FEATURE_NAMES_FILE = os.path.join(MODELS_DIR, "feature_names.json")

# --- Simulation Parameters ---
CLICK_PROBABILITY_PARAM = 0.9

# --- Training Parameters ---
GRID_SEARCH_PARAMS = {'C': [0.1, 1, 10, 100]}

LOINC_TEXT_COLUMNS = ['long_common_name', 'component', 'system', 'property']

# --- MAPPING DICTIONARIES ---
SYSTEM_MAP = {
    'bld': 'blood', 'plas': 'plasma', 'ser': 'serum',
    'ur': 'urine', 'csf': 'cerebrospinal fluid'
}
PROPERTY_MAP = {
    # concentrations
    'mcnc': 'mass concentration','scnc': 'substance concentration','mscnc': 'mass or substance concentration','acnc': 'arbitrary concentration',
    'ccnc': 'catalytic concentration','ncnc': 'number concentration','mfr': 'mass fraction','vfr': 'volume fraction',
    'nfr': 'number fraction','mcrto': 'mass ratio','prthr': 'presence or threshold','type': 'type',
    'susc': 'susceptibility','imp': 'impression/interpretation','temp': 'temperature','num': 'number','cnt': 'count','titr': 'titer',
}