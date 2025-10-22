import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import json

# --- 1. CONFIGURATION ---
#XLS_FILE = '../data/loinc_dataset-v2.xlsx'
XLS_FILE = '../data/expanded_loinc_dataset.xlsx'
OUTPUT_DIR = "../data/feature_rankings"
LOINC_TEXT_COLUMNS = ['long_common_name', 'component', 'system', 'property']
SYSTEM_MAP = {'bld': 'blood', 'plas': 'plasma', 'ser': 'serum', 'ur': 'urine', 'csf': 'cerebrospinal fluid'}
PROPERTY_MAP = {'mcnc': 'mass concentration', 'scnc': 'substance concentration', 'cnt': 'count', 'titr': 'titer',
                'prthr': 'presence or threshold'}


# --- 2. SUPPORT FUNCTION ---

def normalize_loinc_code(code_string, mapping_dict):
    if not isinstance(code_string, str) or code_string.strip() == '':
        return ""
    parts = code_string.lower().split('/')
    translated_parts = [mapping_dict.get(p.strip(), p.strip()) for p in parts]
    return " ".join(translated_parts)


def extract_individual_features(df, query_text):
    """Compute and add the granular feature  (for each column) to DataFrame."""

    df_calc = df.copy()
    df_calc['system'] = df_calc['system'].apply(lambda x: normalize_loinc_code(x, SYSTEM_MAP))
    df_calc['property'] = df_calc['property'].apply(lambda x: normalize_loinc_code(x, PROPERTY_MAP))
    df_features = df.copy()
    query_terms = query_text.lower().split()

    for col in LOINC_TEXT_COLUMNS:
        df_features[col] = df_features[col].fillna('').astype(str)
        df_calc[col] = df_calc[col].fillna('').astype(str)
        df_features[f'{col}_query_terms_count'] = df_calc[col].str.lower().apply(
            lambda text: sum(1 for term in query_terms if term in text)
        )
        try:
            corpus = [query_text] + df_calc[col].tolist()
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            df_features[f'{col}_similarity'] = similarities
        except ValueError:
            df_features[f'{col}_similarity'] = 0.0

    return df_features


def get_general_similarity(df_with_features, query_text):
    """Compute a general similarity score for ranking."""
    df_calc = df_with_features.copy()
    df_calc['system'] = df_calc['system'].apply(lambda x: normalize_loinc_code(x, SYSTEM_MAP))
    df_calc['property'] = df_calc['property'].apply(lambda x: normalize_loinc_code(x, PROPERTY_MAP))


    doc_texts = df_calc.apply(lambda row: " ".join([
        str(row.get('long_common_name', '')),
        str(row.get('component', '')),
        str(row.get('system', '')),
        str(row.get('property', ''))
    ]).strip(), axis=1)

    corpus = [query_text] + doc_texts.tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = tfidf_matrix[0]
    doc_vecs = tfidf_matrix[1:]

    return cosine_similarity(query_vec, doc_vecs)[0]


# --- 3. (MAIN) ---

def main():
    try:
        xls = pd.ExcelFile(XLS_FILE)
    except FileNotFoundError:
        print(f"ERRORE: File '{XLS_FILE}' non trovato.")
        return
    sheets = xls.sheet_names
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    for sheet in sheets:

        df_raw = xls.parse(sheet, header=None)

        # 1. Extract query and data
        raw_query = df_raw.iat[0, 0]
        query_text = re.sub(r'Query:\s*', '', str(raw_query)).strip()
        header_row_index = df_raw[df_raw[0].str.contains('loinc_num', na=False, case=False)].index[0]
        df_data = df_raw.iloc[header_row_index + 1:].reset_index(drop=True)
        df_data.columns = ['loinc_num', 'long_common_name', 'component', 'system', 'property']
        df_data = df_data.dropna(subset=['loinc_num']).reset_index(drop=True)


        # 2. Compute features
        df_with_features = extract_individual_features(df_data, query_text)

        # 3. Compute base ranking (r)
        general_sim_scores = get_general_similarity(df_with_features, query_text)
        df_with_features['ranking_score_baseline'] = general_sim_scores

        # 4. # Sort and save results
        ranked_df = df_with_features.sort_values(by='ranking_score_baseline', ascending=False)



        safe_filename = query_text.lower().replace(" ", "_") + ".csv"
        output_path = os.path.join(OUTPUT_DIR, safe_filename)

        ranked_df.to_csv(output_path, index=False)





if __name__ == "__main__":
    main()