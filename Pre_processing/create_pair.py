import pandas as pd
import os
import glob

# --- 1. CONFIGURATION ---
# Path to the folder containing the simulated click logs
INPUT_DIR = "../data/click_logs"

# Path to the output file that will contain the conceptual preference pairs
OUTPUT_FILE = "../data/conceptual_preference_pairs.csv"

# --- 2. MAIN FUNCTION ---

def main():
    """
    Orchestrates the creation of the conceptual preference dataset:
    1. Finds all click log files.
    2. For each file, generates preference pairs (preferred_doc > not_preferred_doc).
    3. Saves all pairs in a single readable CSV file.
    """
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input folder '{INPUT_DIR}' not found.")
        return

    click_log_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    if not click_log_files:
        print(f"WARNING: No .csv files found in '{INPUT_DIR}'.")
        return

    all_preference_pairs = []

    # Process each log file (one per query)
    for file_path in click_log_files:
        filename = os.path.basename(file_path)

        # Extract the query text from the file name
        query_text = filename.replace('.csv', '').replace('_', ' ')

        df = pd.read_csv(file_path)

        # Find the indices of the clicked documents
        clicked_indices = df[df['clicked'] == True].index

        if len(clicked_indices) == 0:
            print("  -> No clicks found in this log. No pairs generated.")
            continue

        # Generate preference pairs according to the paper's logic
        pairs_generated_count = 0
        for i in clicked_indices:
            # The clicked (preferred) document
            preferred_doc_id = df.loc[i, 'loinc_num']

            # Iterate over all documents that preceded it in the ranking
            for j in range(i):
                # The document seen but NOT clicked (not preferred)
                if not df.loc[j, 'clicked']:
                    not_preferred_doc_id = df.loc[j, 'loinc_num']

                    # Create a row representing this preference
                    pair_data = {
                        'query': query_text,
                        'preferred_doc_id': preferred_doc_id,
                        'not_preferred_doc_id': not_preferred_doc_id,
                        'label': 1  # Use +1 to indicate the first ID is preferred over the second
                    }
                    all_preference_pairs.append(pair_data)
                    pairs_generated_count += 1


    if not all_preference_pairs:
        print("\n--- No preference pairs were generated from any file. ---")
        return

    # Create the final DataFrame and save it
    df_pairs = pd.DataFrame(all_preference_pairs)

    df_pairs.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
