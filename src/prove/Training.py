import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import re

# 1. Carica file Excel e ottieni tutti i fogli
xls = pd.ExcelFile('../../data/loinc_dataset-v2.xlsx')
sheets = xls.sheet_names
print(f"Fogli trovati: {sheets}")

# 2. Preprocess per ogni foglio
queries_data = {}
for sheet in sheets:
    df = xls.parse(sheet, header=None)
    print(f"\nProcessando foglio '{sheet}':")
    print(f"Dimensioni originali: {df.shape}")

    # Estrai la query dalla prima riga
    raw_q = df.iat[0, 0]
    query_text = re.sub(r'Query:\s*', '', str(raw_q)).strip().lower()
    print(f"Query estratta: '{query_text}'")

    # Trova la riga dell'header (quella con 'loinc_num')
    header_row = None
    for i in range(1, min(5, len(df))):  # cerca nelle prime 5 righe
        if pd.notna(df.iat[i, 0]) and 'loinc_num' in str(df.iat[i, 0]).lower():
            header_row = i
            break

    if header_row is None:
        print(f"ERRORE: Header non trovato nel foglio {sheet}")
        continue

    print(f"Header trovato alla riga {header_row}")

    # Dati iniziano dalla riga successiva all'header
    df_data = df.iloc[header_row + 1:].reset_index(drop=True)
    df_data.columns = ['loinc_num', 'long_common_name', 'component', 'system', 'property']

    # Rimuovi righe vuote o con NaN
    df_data = df_data.dropna(subset=['loinc_num', 'long_common_name']).reset_index(drop=True)

    print(f"Righe dopo pulizia: {len(df_data)}")
    print(f"Prime 3 righe loinc_num: {df_data['loinc_num'].head(3).tolist()}")

    queries_data[query_text] = df_data

# 3. Estrai feature complete da tutte le colonne LOINC
print("\nEstraendo feature...")
loinc_columns = ['long_common_name', 'component', 'system', 'property']

for qid, df in queries_data.items():
    query_terms = qid.split()

    # Feature di base
    df['len_name'] = df['long_common_name'].astype(str).str.len()
    df['word_count'] = df['long_common_name'].astype(str).str.split().str.len()

    # Feature per ogni colonna LOINC
    for col in loinc_columns:
        # Assicurati che la colonna sia stringa
        df[col] = df[col].fillna('').astype(str)

        # Numero di termini della query presenti nel campo
        df[f'{col}_query_terms'] = df[col].str.lower().apply(
            lambda x: sum(1 for term in query_terms if term in x)
        )

        # Lunghezza del campo
        df[f'{col}_len'] = df[col].str.len()

        # Numero di parole nel campo
        df[f'{col}_words'] = df[col].str.split().str.len()

        # Calcola similarità TF-IDF per ogni campo
        try:
            texts = [qid] + df[col].tolist()
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarities = cosine_similarity(tfidf_matrix[1:], tfidf_matrix[0:1]).flatten()
            df[f'{col}_similarity'] = similarities
        except:
            # Se TF-IDF fallisce (testi vuoti), usa similarità zero
            df[f'{col}_similarity'] = 0.0

    print(f"Query '{qid}': feature estratte per {len(df)} documenti")

# 4. Genera coppie pairwise basate sulla similarità long_common_name
print("\nGenerando coppie pairwise...")
pairwise = []
for qid, df in queries_data.items():
    df_sorted = df.sort_values('long_common_name_similarity', ascending=False).reset_index(drop=True)
    for i in range(len(df_sorted)):
        for j in range(i + 1, len(df_sorted)):
            pairwise.append((qid, df_sorted.at[i, 'loinc_num'], df_sorted.at[j, 'loinc_num']))

print(f"Coppie pairwise generate: {len(pairwise)}")

# 5. Definisci feature complete
feature_cols = [
    'len_name', 'word_count',
    'long_common_name_query_terms', 'long_common_name_len', 'long_common_name_words', 'long_common_name_similarity',
    'component_query_terms', 'component_len', 'component_words', 'component_similarity',
    'system_query_terms', 'system_len', 'system_words', 'system_similarity',
    'property_query_terms', 'property_len', 'property_words', 'property_similarity'
]

print(f"Feature utilizzate: {len(feature_cols)}")
for i, feat in enumerate(feature_cols):
    print(f"  {i + 1:2d}. {feat}")

# 6. Prepara X_train, y_train con metadati
X, y = [], []
metadata = []
errors = 0

for qid, di, dj in pairwise:
    df = queries_data[qid]

    di_str = str(di)
    dj_str = str(dj)
    df_loinc_str = df['loinc_num'].astype(str)

    mask_i = df_loinc_str == di_str
    mask_j = df_loinc_str == dj_str

    if not mask_i.any() or not mask_j.any():
        errors += 1
        continue

    try:
        vi = df.loc[mask_i, feature_cols].values.flatten().astype(float)
        vj = df.loc[mask_j, feature_cols].values.flatten().astype(float)

        if len(vi) != len(feature_cols) or len(vj) != len(feature_cols):
            errors += 1
            continue

        X.append(vi - vj)
        y.append(1)

        # Salva metadati della coppia
        doc_i_name = df.loc[mask_i, 'long_common_name'].iloc[0]
        doc_j_name = df.loc[mask_j, 'long_common_name'].iloc[0]
        doc_i_sim = df.loc[mask_i, 'long_common_name_similarity'].iloc[0]
        doc_j_sim = df.loc[mask_j, 'long_common_name_similarity'].iloc[0]

        metadata.append({
            'query': qid,
            'doc_i_id': di_str,
            'doc_j_id': dj_str,
            'doc_i_name': doc_i_name,
            'doc_j_name': doc_j_name,
            'doc_i_similarity': doc_i_sim,
            'doc_j_similarity': doc_j_sim,
            'preference': '+1'
        })
    except Exception as e:
        print(f"Errore nella coppia {di_str}, {dj_str}: {e}")
        errors += 1
        continue

print(f"\nErrori durante estrazione feature: {errors}")
print(f"Coppie valide elaborate: {len(X)}")

if not X:
    print("Nessuna coppia valida trovata!")
    exit()

X_train = np.vstack(X)
y_train = np.array(y)
metadata_df = pd.DataFrame(metadata)

# 7. Salva output con feature estese
pd.DataFrame(X_train, columns=feature_cols).to_csv('../data/X_train.csv', index=False)
pd.DataFrame({'y': y_train}).to_csv('../data/y_train.csv', index=False)
metadata_df.to_csv('training_metadata.csv', index=False)

# File pairwise dettagliato
with open('../data/pairwise_detailed.txt', 'w') as f:
    for i, (qid, di, dj) in enumerate(pairwise):
        if i < len(metadata):
            meta = metadata[i]
            f.write(f"+1 qid:{qid} doc_i:{di} doc_j:{dj} # {meta['doc_i_name'][:50]} > {meta['doc_j_name'][:50]}\n")

# File pairwise semplice (compatibile SVMRank)
with open('../data/pairwise.txt', 'w') as f:
    for qid, di, dj in pairwise:
        f.write(f"+1 qid:{qid} {di} {dj}\n")

# 8. Statistiche finali
print("\n" + "=" * 60)
print("GENERAZIONE COMPLETATA")
print("=" * 60)
print(f"File generati:")
print(f"- X_train.csv: {X_train.shape[0]} coppie × {X_train.shape[1]} feature")
print(f"- y_train.csv: {len(y_train)} etichette")
print(f"- training_metadata.csv: metadati delle coppie")
print(f"- pairwise.txt: formato SVMRank")
print(f"- pairwise_detailed.txt: formato leggibile")

print(f"\nStatistiche dataset:")
print(f"- Queries totali: {len(queries_data)}")
print(f"- Documenti totali: {sum(len(df) for df in queries_data.values())}")
print(f"- Coppie pairwise: {len(pairwise)}")
print(f"- Feature per coppia: {len(feature_cols)}")

print(f"\nFeature utilizzate:")
for col in loinc_columns:
    print(f"- {col}: termini_query, lunghezza, parole, similarità")
print(f"- Feature base: lunghezza nome, conteggio parole")
