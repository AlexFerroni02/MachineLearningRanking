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
# --- 1. CONFIGURAZIONE ---
CONCEPTUAL_PAIRS_FILE = "../data/conceptual_preference_pairs.csv"
FEATURE_RANKINGS_DIR = "../data/feature_rankings"
MODEL_OUTPUT_FILE = "ranking_svm_model_optimized.joblib"
SCALER_OUTPUT_FILE = "feature_scaler.joblib"
FEATURE_NAMES_FILE = "feature_names.json"


# --- 2. FUNZIONI DI SUPPORTO ---

def get_feature_columns(df):
    """Identifica le colonne di feature numeriche (senza '_diff')."""
    exclude_cols = ['loinc_num', 'long_common_name', 'component', 'system', 'property', 'clicked']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    return feature_cols


def load_all_features(feature_dir):
    """Carica e unisce tutte le feature grezze da tutti i file di ranking."""
    all_dfs = []
    for file_path in glob.glob(os.path.join(feature_dir, "*.csv")):
        all_dfs.append(pd.read_csv(file_path))

    # Unisce tutti i documenti in un unico DataFrame e rimuove i duplicati
    df_all_features = pd.concat(all_dfs).drop_duplicates(subset=['loinc_num']).set_index('loinc_num')
    return df_all_features


def create_difference_vectors(pairs_df, features_df, feature_cols, scaler):
    """
    Crea i vettori differenza scalati partendo dalle coppie di preferenza
    e da un scaler GIA ADDESTRATO.
    """
    vectors = []
    labels = []

    for index, row in pairs_df.iterrows():
        try:
            vec_pref = features_df.loc[row['preferred_doc_id'], feature_cols]
            vec_not_pref = features_df.loc[row['not_preferred_doc_id'], feature_cols]

            # Trasforma i vettori grezzi usando lo scaler
            vec_pref_scaled = scaler.transform(vec_pref.values.reshape(1, -1))[0]
            vec_not_pref_scaled = scaler.transform(vec_not_pref.values.reshape(1, -1))[0]

            # Calcola la differenza sui dati scalati
            vectors.append(vec_pref_scaled - vec_not_pref_scaled)
            labels.append(1)

            # Aggiungi la coppia inversa
            vectors.append(vec_not_pref_scaled - vec_pref_scaled)
            labels.append(-1)

        except KeyError as e:
            # Salta la coppia se un ID non viene trovato
            continue

    return pd.DataFrame(vectors, columns=[f"{col}_diff" for col in feature_cols]), pd.Series(labels, name="label")


# --- 3. FUNZIONE PRINCIPALE (MAIN) ---

def main():
    print("--- Inizio Pipeline di Training Avanzata ---")

    # 1. Carica le coppie di preferenza concettuali
    try:
        df_pairs = pd.read_csv(CONCEPTUAL_PAIRS_FILE)
    except FileNotFoundError:
        print(f"ERRORE: File '{CONCEPTUAL_PAIRS_FILE}' non trovato.")
        return
    print(f"Caricate {len(df_pairs)} coppie di preferenza.")

    # 2. Carica TUTTE le feature grezze di TUTTI i documenti
    df_all_features = load_all_features(FEATURE_RANKINGS_DIR)
    feature_columns = get_feature_columns(df_all_features)
    print(f"Caricate le feature grezze per {len(df_all_features)} documenti unici.")

    # 3. Dividi le coppie di preferenza in Train e Test
    # Questo è il passaggio FONDAMENTALE per evitare data leakage
    pairs_train, pairs_test = train_test_split(df_pairs, test_size=0.2, random_state=42)
    print(f"Coppie di preferenza divise: {len(pairs_train)} train, {len(pairs_test)} test.")

    # 4. Addestra lo Scaler SOLO sulle feature dei documenti che appaiono nel set di TRAIN
    train_doc_ids = pd.unique(np.concatenate((pairs_train['preferred_doc_id'], pairs_train['not_preferred_doc_id'])))
    train_features = df_all_features.loc[train_doc_ids, feature_columns]

    print("Addestramento dello StandardScaler SOLO sui dati di training...")
    scaler = StandardScaler()
    scaler.fit(train_features)
    print("Scaler addestrato.")

    # 5. Crea i vettori differenza SCALATI per il training e il test
    print("Creazione dei vettori differenza per il set di training...")
    X_train, y_train = create_difference_vectors(pairs_train, df_all_features, feature_columns, scaler)
    print(f"Creati {len(X_train)} esempi di training.")

    print("Creazione dei vettori differenza per il set di test...")
    X_test, y_test = create_difference_vectors(pairs_test, df_all_features, feature_columns, scaler)
    print(f"Creati {len(X_test)} esempi di test.")

    # 6. Esegui GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100]}
    svm = SVC(kernel='linear')
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

    print("Avvio GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # 7. Ottieni e valuta il modello migliore
    print("\nGridSearchCV completato.")
    print(f"Miglior C trovato: {grid_search.best_params_['C']}")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuratezza finale del modello sul test set: {accuracy:.2f}")

    # 8. Salva modello, scaler e nomi delle feature
    joblib.dump(best_model, MODEL_OUTPUT_FILE)
    joblib.dump(scaler, SCALER_OUTPUT_FILE)
    # Salva i nomi delle feature GREZZE (senza _diff)
    with open(FEATURE_NAMES_FILE, 'w') as f:
        json.dump(feature_columns, f)

    print(f"Modello, Scaler e Nomi Feature salvati correttamente.")

    # 9. Analizza i pesi del modello
    print("\n--- Pesi del Modello (Feature Importance) ---")
    weights = best_model.coef_[0]
    df_weights = pd.DataFrame({'feature': X_train.columns, 'weight': weights})
    print(df_weights.sort_values(by='weight', ascending=False))


if __name__ == "__main__":
    main()