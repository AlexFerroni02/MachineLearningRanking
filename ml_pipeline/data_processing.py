import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import json


# --- Helper Functions (Internal to this module) ---

def normalize_loinc_code(code_string, mapping_dict):
    """Normalizes LOINC codes using the provided dictionary."""
    if not isinstance(code_string, str) or code_string.strip() == '': return ""
    parts = code_string.lower().split('/')
    translated_parts = [mapping_dict.get(p.strip(), p.strip()) for p in parts]
    return " ".join(translated_parts)


def extract_individual_features(df, query_text, system_map, property_map, loinc_text_columns):
    """
    Calculates and adds granular features (one per column) to the DataFrame.
    Uses provided maps for normalization.
    """
    print("  -> Calculating individual features...")
    df_calc = df.copy()  # Temporary copy for calculations with normalized text
    # Apply normalization using maps passed as arguments
    df_calc['system'] = df_calc['system'].apply(lambda x: normalize_loinc_code(x, system_map))
    df_calc['property'] = df_calc['property'].apply(lambda x: normalize_loinc_code(x, property_map))

    df_features = df.copy()  # Start with the original df to add features to
    query_terms = query_text.lower().split()

    for col in loinc_text_columns:
        # Clean original data (for output) and normalized data (for calculation)
        df_features[col] = df_features[col].fillna('').astype(str)
        df_calc[col] = df_calc[col].fillna('').astype(str)  # Ensure df_calc is also clean

        # Term count uses the normalized version from df_calc
        df_features[f'{col}_query_terms_count'] = df_calc[col].str.lower().apply(
            lambda text: sum(1 for term in query_terms if term in text)
        )

        # Similarity uses the normalized version from df_calc
        try:
            corpus = [query_text] + df_calc[col].tolist()  # Use normalized text
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            # Add feature to the original DataFrame
            df_features[f'{col}_similarity'] = similarities
        except ValueError:
            df_features[f'{col}_similarity'] = 0.0

    print("  -> Individual features calculated.")
    return df_features


def get_general_similarity(df_with_features, query_text, system_map, property_map):
    """
    Calculates a "general" similarity score by combining text fields.
    Uses provided maps for normalization. This is ONLY for the baseline rank.
    """
    print("  -> Calculating 'general' similarity for baseline ranking...")
    df_calc = df_with_features.copy()
    # Apply normalization using maps passed as arguments
    df_calc['system'] = df_calc['system'].apply(lambda x: normalize_loinc_code(x, system_map))
    df_calc['property'] = df_calc['property'].apply(lambda x: normalize_loinc_code(x, property_map))

    # Combine normalized text fields
    doc_texts = df_calc.apply(lambda row: " ".join([
        str(row.get('component', '')),
        str(row.get('system', '')),  # Use normalized text
        str(row.get('property', ''))  # Use normalized text
    ]).strip(), axis=1)

    corpus = [query_text] + doc_texts.tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = tfidf_matrix[0]
    doc_vecs = tfidf_matrix[1:]

    return cosine_similarity(query_vec, doc_vecs)[0]


# --- Main Function (to be imported by train.py) ---

def run_feature_engineering(xls_file_path, output_dir, system_map, property_map, loinc_text_columns):
    """
    Orchestrates the feature engineering process.
    This is the function your 'train.py' will import and call.
    """
    try:
        xls = pd.ExcelFile(xls_file_path)
    except FileNotFoundError:
        print(f"ERROR: File '{xls_file_path}' not found.")
        return False  # Indicate failure

    sheets = xls.sheet_names
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    print(f"\n--- Starting Feature Engineering Phase ---")
    for sheet in sheets:
        print(f"\nProcessing sheet: '{sheet}'...")
        df_raw = xls.parse(sheet, header=None)

        # 1. Extract query and raw data
        raw_query = df_raw.iat[0, 0]
        query_text = re.sub(r'Query:\s*', '', str(raw_query)).strip()

        # Find header row robustly
        header_row_indices = df_raw[df_raw[0].str.contains('loinc_num', na=False, case=False)].index
        if not header_row_indices.any():
            print(f"  -> WARNING: Header 'loinc_num' not found in sheet '{sheet}'. Skipping.")
            continue
        header_row_index = header_row_indices[0]

        df_data = df_raw.iloc[header_row_index + 1:].reset_index(drop=True)
        # Ensure correct column names even if header has extra cols
        df_data.columns = ['loinc_num', 'long_common_name', 'component', 'system', 'property'][:len(df_data.columns)]
        df_data = df_data.dropna(subset=['loinc_num']).reset_index(drop=True)
        print(f"  Query extracted: '{query_text}'. Found {len(df_data)} documents.")

        # 2. Extract ALL individual features
        # Pass the maps to the helper function
        df_with_features = extract_individual_features(df_data, query_text, system_map, property_map,
                                                       loinc_text_columns)

        # 3. Calculate the "naive" baseline ranking score (r)
        general_sim_scores = get_general_similarity(df_with_features, query_text, system_map, property_map)
        df_with_features['ranking_score_baseline'] = general_sim_scores

        # Sort by this baseline score to create the initial ranked list
        ranked_df = df_with_features.sort_values(by='ranking_score_baseline', ascending=False)
        print("  -> Baseline 'general' ranking created.")

        # 4. Save the result (contains ORIGINAL data + ALL features)
        safe_filename = query_text.lower().replace(" ", "_") + ".csv"
        output_path = os.path.join(output_dir, safe_filename)

        ranked_df.to_csv(output_path, index=False)
        print(f"  -> Results saved (with ALL features) to: '{output_path}'")

    print(f"\n--- Feature Engineering Phase Completed! ---")
    return True  # Indicate success