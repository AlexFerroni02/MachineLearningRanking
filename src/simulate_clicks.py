import pandas as pd
import numpy as np
import os
import glob

# --- 1. CONFIGURAZIONE ---
# Percorso della cartella contenente i ranking con le feature
INPUT_DIR = "../data/feature_rankings"

# Percorso della nuova cartella dove salveremo i risultati con i click simulati
OUTPUT_DIR = "../data/click_logs"

# Parametro per il modello di click. Un valore più alto rende i click più probabili.
CLICK_PROBABILITY_PARAM = 0.8


# --- 2. FUNZIONE DI SUPPORTO ---

def simulate_clicks_for_ranking(df, prob_param):
    """
    Aggiunge una colonna 'clicked' a un DataFrame di ranking basato su un modello probabilistico.
    La probabilità di click diminuisce con la posizione nel ranking.
    """
    print(f"  -> Simulando i click per {len(df)} documenti...")

    click_decisions = []

    # Itera su ogni documento usando il suo indice come posizione nel ranking (i)
    for i, row in df.iterrows():
        # Calcola la probabilità di click per questa posizione
        # La formula P(click) = C / (i + 1) è un modello comune per la simulazione
        # 'i' è l'indice della riga, che parte da 0 per il primo risultato.
        click_prob = prob_param / (i + 1)

        # "Lancia il dado": genera un numero casuale tra 0 e 1
        random_chance = np.random.rand()

        # Se il nostro "lancio" è inferiore alla probabilità, l'utente ha cliccato
        if random_chance < click_prob:
            click_decisions.append(True)
        else:
            click_decisions.append(False)

    # Aggiunge la lista di decisioni come una nuova colonna al DataFrame
    df_with_clicks = df.copy()
    df_with_clicks['clicked'] = click_decisions

    print(f"  -> Simulazione completata. Totale click generati: {sum(click_decisions)}.")

    return df_with_clicks


def simulate_clicks_advanced(df, prob_param):
    """
    Simula i click con un modello a cascata avanzato che considera
    la posizione, la rilevanza e la soddisfazione dell'utente.
    """
    print(f"  -> Simulando i click con il modello avanzato per {len(df)} documenti...")

    click_decisions = []
    is_user_satisfied = False

    # La colonna di rilevanza da usare (la nostra baseline)
    relevance_score_col = 'long_common_name_similarity'

    for i, row in df.iterrows():
        clicked_this_doc = False

        if not is_user_satisfied:
            # Calcola la probabilità di click combinando posizione e rilevanza
            relevance_score = row[relevance_score_col]
            prob_click = (prob_param / (i + 1)) * relevance_score

            # Simula la decisione di click
            if np.random.rand() < prob_click:
                clicked_this_doc = True

                # Se l'utente clicca, c'è una possibilità che sia soddisfatto e smetta di cercare
                # La probabilità di essere soddisfatto è legata alla rilevanza del documento
                prob_satisfaction = relevance_score
                if np.random.rand() < prob_satisfaction:
                    is_user_satisfied = True

        click_decisions.append(clicked_this_doc)

    # Aggiunge la lista di decisioni come una nuova colonna
    df_with_clicks = df.copy()
    df_with_clicks['clicked'] = click_decisions

    print(f"  -> Simulazione avanzata completata. Totale click generati: {sum(click_decisions)}.")

    return df_with_clicks


# --- 3. FUNZIONE PRINCIPALE (MAIN) ---

def main():
    """
    Orchestra il processo:
    1. Trova tutti i file di ranking.
    2. Per ogni file, simula i click.
    3. Salva un nuovo file con la colonna 'clicked' aggiuntiva.
    """
    # Controlla se la cartella di input esiste
    if not os.path.exists(INPUT_DIR):
        print(f"ERRORE: La cartella di input '{INPUT_DIR}' non è stata trovata.")
        return

    # Crea la cartella di output se non esiste
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Cartella di output '{OUTPUT_DIR}' creata.")

    # Trova tutti i file .csv nella cartella di input
    ranking_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    if not ranking_files:
        print(f"ATTENZIONE: Nessun file .csv trovato in '{INPUT_DIR}'.")
        return

    print(f"--- Trovati {len(ranking_files)} file di ranking da processare ---")

    # Itera su ogni file di ranking
    for file_path in ranking_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessando il file: '{filename}'...")

        # 1. Carica il file CSV
        df_ranking = pd.read_csv(file_path)

        # 2. Simula i click per questo ranking
        df_with_clicks = simulate_clicks_for_ranking(df_ranking, CLICK_PROBABILITY_PARAM)
        # con la simulazione avanzata
        #df_with_clicks = simulate_clicks_advanced(df_ranking, CLICK_PROBABILITY_PARAM)
        # 3. Salva il nuovo file nella cartella di output
        output_path = os.path.join(OUTPUT_DIR, filename)
        df_with_clicks.to_csv(output_path, index=False)


    print(f"\n--- Processo di simulazione completato! Controlla la cartella '{OUTPUT_DIR}'. ---")


if __name__ == "__main__":
    main()