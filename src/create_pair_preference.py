import pandas as pd
import os
import glob

# --- 1. CONFIGURAZIONE ---
# Percorso della cartella contenente i log con i click simulati
INPUT_DIR = "../data/click_logs"

# Percorso del file di output che conterrà le coppie di preferenza concettuali
OUTPUT_FILE = "../data/conceptual_preference_pairs.csv"


# --- 2. FUNZIONE PRINCIPALE (MAIN) ---

def main():
    """
    Orchestra la creazione del dataset di preferenze concettuali:
    1. Trova tutti i file di click log.
    2. Per ogni file, genera le coppie di preferenza (doc_preferito > doc_non_preferito).
    3. Salva tutte le coppie in un unico file CSV leggibile.
    """
    if not os.path.exists(INPUT_DIR):
        print(f"ERRORE: La cartella di input '{INPUT_DIR}' non è stata trovata.")
        return

    click_log_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    if not click_log_files:
        print(f"ATTENZIONE: Nessun file .csv trovato in '{INPUT__DIR}'.")
        return

    print(f"--- Trovati {len(click_log_files)} file di click log da processare ---")

    all_preference_pairs = []

    # Processa ogni file di log (uno per query)
    for file_path in click_log_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessando il file: '{filename}'...")

        # Estrai il testo della query dal nome del file
        query_text = filename.replace('.csv', '').replace('_', ' ')

        df = pd.read_csv(file_path)

        # Trova gli indici dei documenti cliccati
        clicked_indices = df[df['clicked'] == True].index

        if len(clicked_indices) == 0:
            print("  -> Nessun click trovato in questo log. Nessuna coppia generata.")
            continue

        # Genera le coppie di preferenza secondo la logica del paper
        pairs_generated_count = 0
        for i in clicked_indices:
            # Il documento cliccato (preferito)
            preferred_doc_id = df.loc[i, 'loinc_num']

            # Itera su tutti i documenti che lo precedevano nel ranking
            for j in range(i):
                # Il documento visto ma NON cliccato (non preferito)
                if not df.loc[j, 'clicked']:
                    not_preferred_doc_id = df.loc[j, 'loinc_num']

                    # Crea una riga che rappresenta questa preferenza
                    pair_data = {
                        'query': query_text,
                        'preferred_doc_id': preferred_doc_id,
                        'not_preferred_doc_id': not_preferred_doc_id,
                        'label': 1  # Usiamo +1 per indicare che il primo ID è preferito al secondo
                    }
                    all_preference_pairs.append(pair_data)
                    pairs_generated_count += 1

        print(f"  -> Generate {pairs_generated_count} coppie di preferenza da questo log.")

    if not all_preference_pairs:
        print("\n--- Nessuna coppia di preferenza è stata generata da alcun file. ---")
        return

    # Crea il DataFrame finale e salvalo
    df_pairs = pd.DataFrame(all_preference_pairs)

    df_pairs.to_csv(OUTPUT_FILE, index=False)

    print(f"\n--- Processo completato! ---")
    print(f"File concettuale con {len(df_pairs)} coppie di preferenza salvato in: '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()