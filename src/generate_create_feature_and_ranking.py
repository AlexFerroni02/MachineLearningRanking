import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# --- 1. CONFIGURATION ---
XLS_FILE = '../data/loinc_dataset-v2.xlsx'
OUTPUT_DIR = "../data/feature_rankings"

# Original columns to read from and use in the final output file
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


# --- 2. HELPER FUNCTIONS ---

def normalize_loinc_code(code_string, mapping_dict):
    """Normalizes LOINC codes, also handling combined cases like 'Bld/Ser'."""
    if not isinstance(code_string, str) or code_string.strip() == '':
        return ""
    parts = code_string.lower().split('/')
    translated_parts = [mapping_dict.get(p.strip(), p.strip()) for p in parts]
    return " ".join(translated_parts)


def extract_features(df, query_text):
    """
    Extracts features. It uses a normalized copy of the data for calculations
    but adds the features to the original DataFrame.
    """

    # --- PREPARING DATA FOR CALCULATION ---
    # 1. Create a temporary copy of the DataFrame to avoid modifying the original.
    df_calc = df.copy()

    # 2. Normalize ONLY the 'system' and 'property' columns in the temporary copy.
    df_calc['system'] = df_calc['system'].apply(lambda x: normalize_loinc_code(x, SYSTEM_MAP))
    df_calc['property'] = df_calc['property'].apply(lambda x: normalize_loinc_code(x, PROPERTY_MAP))

    # --- FEATURE CALCULATION ---
    df_features = df.copy()
    query_terms = query_text.lower().split()

    for col in LOINC_TEXT_COLUMNS:
        # Cleans the data in both the original and the calculation DataFrames
        df_features[col] = df_features[col].fillna('').astype(str)
        df_calc[col] = df_calc[col].fillna('').astype(str)

        # The term count features are calculated on the normalized data
        df_features[f'{col}_query_terms_count'] = df_calc[col].str.lower().apply(
            lambda text: sum(1 for term in query_terms if term in text)
        )

        # The TF-IDF similarity is calculated on the normalized data
        try:
            corpus = [query_text] + df_calc[col].tolist()
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            df_features[f'{col}_similarity'] = similarities
        except ValueError:
            df_features[f'{col}_similarity'] = 0.0


    return df_features


# --- 3. MAIN FUNCTION ---

def main():
    """
    Orchestrates the process:
    1. Load the data.
    2. For each query, call the extract_features function which uses
       temporary normalization to calculate the features.
    3. Save the original DataFrame enriched with the new features.
    """
    try:
        xls = pd.ExcelFile(XLS_FILE)
    except FileNotFoundError:
        print(f"ERROR: File '{XLS_FILE}' not found.")
        return

    sheets = xls.sheet_names

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for sheet in sheets:
        df_raw = xls.parse(sheet, header=None)

        # 1. Extract query and data
        raw_query = df_raw.iat[0, 0]
        query_text = re.sub(r'Query:\s*', '', str(raw_query)).strip()
        header_row_index = df_raw[df_raw[0].str.contains('loinc_num', na=False, case=False)].index[0]
        df_data = df_raw.iloc[header_row_index + 1:].reset_index(drop=True)
        df_data.columns = ['loinc_num', 'long_common_name', 'component', 'system', 'property']
        df_data = df_data.dropna(subset=['loinc_num']).reset_index(drop=True)

        # 2. Extract all features
        # Normalization happens 'under the hood' inside this function
        df_with_features = extract_features(df_data, query_text)

        # 3. Create the baseline ranking
        ranked_df = df_with_features.sort_values(by='long_common_name_similarity', ascending=False)

        # 4. Save the result to a CSV file
        safe_filename = query_text.lower().replace(" ", "_") + ".csv"
        output_path = os.path.join(OUTPUT_DIR, safe_filename)

        ranked_df.to_csv(output_path, index=False)

    print(f"\n--- Process complete! Check the '{OUTPUT_DIR}' folder for the results. ---")


if __name__ == "__main__":
    main()