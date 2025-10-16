import pandas as pd
import numpy as np
import os
import glob

# --- 1. CONFIGURAZIONE ---
# Il file con le coppie di preferenza che abbiamo creato nello script precedente
CONCEPTUAL_PAIRS_FILE = "../data/conceptual_preference_pairs.csv"

# La cartella dove si trovano i file CSV con le feature per ogni query
FEATURE_RANKINGS_DIR = "../data/feature_rankings"

# Il file di output finale, pronto per essere dato in pasto all'SVM
SVM_TRAINING_DATASET_FILE = "../data/svm_training_dataset.csv"


# --- 2. FUNZIONE DI SUPPORTO ---

def get_feature_columns(df):
    """Identifica automaticamente tutte le colonne di feature numeriche."""
    exclude_cols = ['loinc_num', 'long_common_name', 'component', 'system', 'property', 'clicked']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    return feature_cols


# --- 3. FUNZIONE PRINCIPALE (MAIN) ---

def main():
    """
    Orchestra la trasformazione finale del dataset:
    1. Carica le coppie di preferenza concettuali.
    2. Raggruppa le coppie per query per efficienza.
    3. Per ogni query, carica le feature e calcola i vettori differenza.
    4. Salva il dataset di addestramento finale.
    """
    try:
        df_pairs = pd.read_csv(CONCEPTUAL_PAIRS_FILE)
    except FileNotFoundError:
        print(f"ERRORE: File delle coppie concettuali '{CONCEPTUAL_PAIRS_FILE}' non trovato.")
        return

    print(f"--- Caricate {len(df_pairs)} coppie di preferenza da processare ---")

    all_training_vectors = []

    # Raggruppiamo per query. Questo ci permette di caricare ogni file di feature una sola volta.
    for query_text, group in df_pairs.groupby('query'):
        print(f"\nProcessando coppie per la query: '{query_text}'...")

        # Costruisci il percorso al file di feature corrispondente
        safe_filename = query_text.lower().replace(" ", "_") + ".csv"
        feature_file_path = os.path.join(FEATURE_RANKINGS_DIR, safe_filename)

        try:
            df_features = pd.read_csv(feature_file_path)
            # Usiamo loinc_num come indice per ricerche velocissime (lookup)
            df_features.set_index('loinc_num', inplace=True)
        except FileNotFoundError:
            print(f"  -> ATTENZIONE: File di feature '{feature_file_path}' non trovato. Salto questa query.")
            continue

        feature_columns = get_feature_columns(df_features)
        if not feature_columns:
            print("  -> ATTENZIONE: Nessuna colonna di feature trovata nel file. Salto questa query.")
            continue

        # Itera su ogni coppia di preferenza per questa query
        for index, row in group.iterrows():
            pref_id = row['preferred_doc_id']
            not_pref_id = row['not_preferred_doc_id']

            try:
                # Estrai i vettori di feature numeriche per entrambi i documenti
                vec_pref = df_features.loc[pref_id, feature_columns]
                vec_not_pref = df_features.loc[not_pref_id, feature_columns]

                # Calcola il vettore differenza
                diff_vector = vec_pref.values - vec_not_pref.values
                all_training_vectors.append(np.append(diff_vector, 1))  # Aggiungi label +1

                # Aggiungi anche la coppia inversa per un addestramento più robusto
                inverse_diff = vec_not_pref.values - vec_pref.values
                all_training_vectors.append(np.append(inverse_diff, -1))  # Aggiungi label -1

            except KeyError as e:
                print(f"  -> ATTENZIONE: ID documento {e} non trovato nel file di feature. Salto questa coppia.")

    if not all_training_vectors:
        print("\n--- Nessun vettore di addestramento è stato generato. ---")
        return

    # Crea il DataFrame finale pronto per l'SVM
    feature_diff_columns = [f"{col}_diff" for col in feature_columns]
    final_columns = feature_diff_columns + ['label']

    df_training = pd.DataFrame(all_training_vectors, columns=final_columns)

    df_training.to_csv(SVM_TRAINING_DATASET_FILE, index=False)

    print(f"\n--- Processo completato! ---")
    print(f"Dataset di addestramento finale con {len(df_training)} esempi salvato in: '{SVM_TRAINING_DATASET_FILE}'")


if __name__ == "__main__":
    main()