import pandas as pd
import joblib
import numpy as np
import os
import json

# --- CONFIGURAZIONE ---
MODEL_FILE = "ranking_svm_model_optimized.joblib"
SCALER_FILE = "feature_scaler.joblib"
FEATURE_NAMES_FILE = "feature_names.json"
EXAMPLE_FEATURE_FILE = "../data/feature_rankings/glucose_in_blood.csv"


def rank_documents(model, scaler, documents_df, feature_names):
    """
    Usa un modello SVM, uno scaler e una lista di feature
    per calcolare i punteggi di ranking.
    """
    print("\n--- Ranking Documents with Trained Model and Scaler ---")

    # 1. 'feature_names' ora contiene i nomi delle feature grezze (es. "component_similarity")
    feature_cols = feature_names

    if not all(col in documents_df.columns for col in feature_cols):
        print("ERROR: Document file is missing required feature columns.")
        return None

    # 2. Estrai SOLO le feature numeriche
    doc_features = documents_df[feature_cols]

    # 3. APPLICA LO SCALER (usa .transform())
    # Questo ora funziona perché lo scaler è stato addestrato su feature grezze
    doc_features_scaled = scaler.transform(doc_features)
    print("Features successfully scaled using the loaded scaler.")

    # 4. Estrai il vettore dei pesi 'w'
    w = model.coef_[0]

    # 5. Calcola i punteggi SUI DATI SCALATI
    ranking_scores = np.dot(doc_features_scaled, w)

    # Aggiungi i punteggi e ordina
    ranked_df = documents_df.copy()
    ranked_df['ranking_score'] = ranking_scores
    ranked_df = ranked_df.sort_values(by='ranking_score', ascending=False).reset_index(drop=True)

    return ranked_df


def main():
    """
    Carica un modello, uno scaler e i nomi delle feature e li usa per fare ranking.
    """
    # 1. Carica modello, scaler e nomi delle feature
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        with open(FEATURE_NAMES_FILE, 'r') as f:
            feature_names = json.load(f)
        print(f"Successfully loaded model, scaler, and feature names.")
    except FileNotFoundError as e:
        print(f"ERROR: Missing file. Could not load model, scaler, or feature names. {e}")
        return

    # 2. Carica i documenti da classificare
    try:
        docs_to_rank_df = pd.read_csv(EXAMPLE_FEATURE_FILE)
    except FileNotFoundError:
        print(f"ERROR: Document feature file '{EXAMPLE_FEATURE_FILE}' not found.")
        return

    # 3. Ottieni il nuovo ranking
    final_ranking = rank_documents(model, scaler, docs_to_rank_df, feature_names)

    if final_ranking is not None:
        print("\nTop 5 ranked documents based on the learned model:")
        print(final_ranking[['loinc_num', 'long_common_name', 'ranking_score']].head(5))


if __name__ == "__main__":
    main()