import pandas as pd
import numpy as np
import os
import glob

# --- Helper Simulation Function (Internal) ---
def _simulate_clicks_simple_positional(df, prob_param):
    """
    Simulates clicks based *only* on the document's rank (position).
    Probability decreases with rank: P(click) = C / (rank + 1).
    """
    click_decisions = []
    for i, row in df.iterrows():
        click_prob = prob_param / (i + 2)
        if np.random.rand() < click_prob:
            click_decisions.append(True)
        else:
            click_decisions.append(False)
    df_with_clicks = df.copy()
    df_with_clicks['clicked'] = click_decisions
    # Reduced print statement
    print(f"  -> Clicks simulated. Total: {sum(click_decisions)}.")
    return df_with_clicks

# --- Main Function (to be imported by train.py) ---

def run_click_simulation(input_dir, output_dir, prob_param=0.8):
    """
    Orchestrates the simple click simulation process. Reads ranked files,
    simulates clicks positionally, and saves new files with click data.

    Args:
        input_dir (str): Path to the directory with ranked feature files.
        output_dir (str): Path to the directory where click logs will be saved.
        prob_param (float): The base probability parameter for clicks (e.g., 0.8).
    """
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' not found.")
        return False # Indicate failure

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ranking_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not ranking_files:
        print(f"WARNING: No .csv files found in '{input_dir}'.")
        return False # Indicate failure

    print(f"\n--- Starting Click Simulation Phase (Simple Positional Model) ---")
    print(f"Found {len(ranking_files)} files to process.")

    for file_path in ranking_files:
        filename = os.path.basename(file_path)

        try:
            df_ranking = pd.read_csv(file_path)
        except Exception as e:
            print(f"  -> ERROR loading file {filename}: {e}. Skipping.")
            continue

        # Use the simple positional simulation function
        df_with_clicks = _simulate_clicks_simple_positional(df_ranking, prob_param)

        # Save the new file with the 'clicked' column
        output_path = os.path.join(output_dir, filename)
        df_with_clicks.to_csv(output_path, index=False)


    print(f"--- Click Simulation Phase Completed! Check '{output_dir}'. ---")
    return True # Indicate success