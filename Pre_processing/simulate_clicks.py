import pandas as pd
import numpy as np
import os
import glob

# --- 1. CONFIGURATION ---
# Path to the folder containing the feature rankings
INPUT_DIR = "../data/feature_rankings"

# Path to the folder where the simulated click logs will be saved
OUTPUT_DIR = "../data/click_logs"

# Parameter for the click model. Higher values make clicks more likely.
CLICK_PROBABILITY_PARAM = 0.8

MAX_POSITION = 20

# --- 2. SUPPORT FUNCTION ---

def simulate_clicks_for_ranking(df, prob_param):
    """
    Adds a 'clicked' column to a ranking DataFrame using a probabilistic model.
    The click probability decreases with the position in the ranking.
    """
    click_decisions = []

    # Iterate over each document using its index as the ranking position
    for i, row in df.iterrows():
        if i >= MAX_POSITION:
            click_decisions.append(False)
            continue
        # Calculate the click probability for this position
        # Formula: P(click) = C / (i + 1), a common model for simulation
        click_prob = prob_param / (i + 1)

        #generate a random number between 0 and 1
        random_chance = np.random.rand()

        if random_chance < click_prob:
            click_decisions.append(True)
        else:
            click_decisions.append(False)

    # Add the list of decisions as a new column to the DataFrame
    df_with_clicks = df.copy()
    df_with_clicks['clicked'] = click_decisions

    return df_with_clicks

# --- 3. MAIN FUNCTION ---

def main():
    """
    Orchestrates the process:
    1. Finds all ranking files.
    2. For each file, simulates clicks.
    3. Saves a new file with the additional 'clicked' column.
    """
    # Check if the input folder exists
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input folder '{INPUT_DIR}' not found.")
        return

    # Create the output folder if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Output folder '{OUTPUT_DIR}' created.")

    # Find all .csv files in the input folder
    ranking_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    if not ranking_files:
        print(f"WARNING: No .csv files found in '{INPUT_DIR}'.")
        return

    # Iterate over each ranking file
    for file_path in ranking_files:
        filename = os.path.basename(file_path)

        # 1. Load the CSV file
        df_ranking = pd.read_csv(file_path)

        # 2. Simulate clicks for this ranking
        df_with_clicks = simulate_clicks_for_ranking(df_ranking, CLICK_PROBABILITY_PARAM)

        # 3. Save the new file in the output folder
        output_path = os.path.join(OUTPUT_DIR, filename)
        df_with_clicks.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
